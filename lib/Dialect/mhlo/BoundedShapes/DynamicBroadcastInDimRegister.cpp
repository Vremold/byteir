//===- DynamicPartitionRegister.h -----------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/BoundedShapes/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

void mlir::registerDynamicBroadcastInDimInferBoundedReturnTypes() {
  static InferBoundedReturnTypesRegistration shapeRegister(
      mhlo::DynamicBroadcastInDimOp::getOperationName(),
      [](MLIRContext *context, Optional<Location>, ValueRange operands,
         DictionaryAttr, RegionRange,
         SmallVectorImpl<Type> &inferredReturnTypes) {
        auto inputType = operands[0].getType().dyn_cast<RankedTensorType>();
        if (inputType == nullptr) {
          return failure();
        }
        auto dynamicShapeValue = operands[1];
        auto attribute =
            dynamicShapeValue.getType().cast<RankedTensorType>().getEncoding();
        if (attribute && attribute.dyn_cast<DictionaryAttr>()) {
          auto dictAttr = attribute.dyn_cast<DictionaryAttr>();
          auto boundedShapeDense = dictAttr.get(getBoundedShapeDenseAttrName());
          if (boundedShapeDense == nullptr ||
              boundedShapeDense.dyn_cast<DenseIntElementsAttr>() == nullptr) {
            return failure();
          }
          auto boundedShape =
              llvm::to_vector<6>(boundedShapeDense.cast<DenseIntElementsAttr>()
                                     .getValues<int64_t>());
          int64_t leadingDimSize =
              inputType.getRank() - static_cast<int64_t>(boundedShape.size());
          if (leadingDimSize > 0) {
            boundedShape.insert(boundedShape.begin(), leadingDimSize, 1);
          }
          auto inputShape = inputType.getShape();
          size_t placeOffset = boundedShape.size() - inputShape.size();
          for (size_t i = 0; i < boundedShape.size(); ++i) {
            if (i >= placeOffset) {
              boundedShape[i] =
                  std::max(boundedShape[i], inputShape[i - placeOffset]);
            }
          }
          inferredReturnTypes.push_back(RankedTensorType::get(
              boundedShape, IntegerType::get(context, 64)));
          return success();
        }
        return failure();
      });
}
