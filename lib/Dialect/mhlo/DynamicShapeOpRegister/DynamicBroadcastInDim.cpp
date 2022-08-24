//===- DynamicBroadcastInDim.cpp ------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/BoundedShapes/Register.h"
#include "byteir/Dialect/mhlo/ReifyShapes/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

void mlir::registerDynamicBroadcastInDimReifyReturnTypeShapes() {
  static ReifyReturnTypeShapesRegistration shapeRegister(
      mhlo::DynamicBroadcastInDimOp::getOperationName(),
      [](Operation *op, OpBuilder &builder, ValueRange,
         SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) {
        Value dynamicShape = op->getOperand(1);
        if (auto type = dynamicShape.getType().dyn_cast<RankedTensorType>()) {
          SmallVector<Value> dims;
          for (int64_t i = 0; i < type.getRank(); ++i) {
            dims.push_back(builder.create<shape::GetExtentOp>(op->getLoc(),
                                                              dynamicShape, i));
          }
          reifiedReturnShapes.push_back(
              builder.create<tensor::FromElementsOp>(op->getLoc(), dims));
          return success();
        }
        return failure();
      });
}

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
