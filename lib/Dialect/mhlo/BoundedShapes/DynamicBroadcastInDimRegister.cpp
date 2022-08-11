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

void mlir::registerDynamicBroadcastInDimInferBoundedReturnTypeComponents() {
  static InferBoundedReturnTypeComponentsRegistration shapeRegister(
      mhlo::DynamicBroadcastInDimOp::getOperationName(),
      [](MLIRContext *context, Optional<Location>, ValueShapeRange operands,
         DictionaryAttr, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        auto inputType = operands[0].getType().dyn_cast<RankedTensorType>();
        if (inputType == nullptr) {
          return failure();
        }

        ShapeAdaptor dynamicShapeAdaptor = operands.getValueAsShape(1);
        if (!dynamicShapeAdaptor)
          return failure();

        ShapedTypeComponents dynamicShape;
        dynamicShapeAdaptor.getDims(dynamicShape);

        auto boundedShape = llvm::to_vector<6>(dynamicShape.getDims());

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

        Type type =
            RankedTensorType::get(boundedShape, IntegerType::get(context, 64));
        inferredReturnTypes.push_back(type.cast<ShapedType>());
        return success();
      });
}
