//===- ReduceRegister.cpp -------------------------------------*--- C++ -*-===//
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

// TODO: this should be removed when push to upstream
void mlir::registerReduceInferBoundedReturnTypeComponents() {
  static InferBoundedReturnTypeComponentsRegistration shapeRegister(
      mhlo::ReduceOp::getOperationName(),
      [](MLIRContext *context, Optional<Location>, ValueShapeRange operands,
         DictionaryAttr attr, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        auto inputType = operands[0].getType().dyn_cast<RankedTensorType>();
        if (inputType == nullptr || !inputType.hasStaticShape()) {
          return failure();
        }
        auto dimensions =
            attr.get("dimensions").dyn_cast<DenseIntElementsAttr>();
        if (dimensions == nullptr) {
          return failure();
        }
        int64_t rank = inputType.getRank();
        llvm::SmallVector<bool, 4> dimsMask(rank, false);
        for (int64_t dim : dimensions.getValues<int64_t>())
          dimsMask[dim] = true;

        SmallVector<int64_t, 4> shape;
        for (int64_t i = 0; i < rank; ++i) {
          if (!dimsMask[i])
            shape.push_back(inputType.getDimSize(i));
        }
        Type type = RankedTensorType::get(shape, IntegerType::get(context, 64));
        inferredReturnTypes.push_back(type.cast<ShapedType>());
        return success();
      });
}
