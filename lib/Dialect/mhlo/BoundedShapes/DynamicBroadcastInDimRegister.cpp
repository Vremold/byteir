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
         DictionaryAttr attr, RegionRange,
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
        auto bcastDimensions = attr.get("broadcast_dimensions")
                                   .dyn_cast_or_null<DenseIntElementsAttr>();
        if (bcastDimensions == nullptr) {
          return failure();
        }
        auto bcastDimensionsType = bcastDimensions.getType();

        auto bcastDimensionsSize = bcastDimensionsType.getNumElements();

        auto boundedShape = llvm::to_vector<6>(dynamicShape.getDims());
        for (int i = 0; i != bcastDimensionsSize; ++i) {
          auto dimIndex = bcastDimensions.getValues<int64_t>()[i];
          boundedShape[dimIndex] =
              std::max(boundedShape[dimIndex], inputType.getShape()[i]);
        }
        Type type =
            RankedTensorType::get(boundedShape, IntegerType::get(context, 64));
        inferredReturnTypes.push_back(type.cast<ShapedType>());
        return success();
      });
}
