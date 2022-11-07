//===- DynamicPartition.h -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

void mlir::registerDynamicPartitionShapeConstraints() {
  static InsertShapeConstraintRegistration shapeRegister(
      getDynamicPartitionName(), [](Operation *op, OpBuilder &builder) {
        // init builder position
        builder.setInsertionPointAfter(op);
        SmallVector<Value> dim0OfResults;
        for (Value res : op->getResults()) {
          dim0OfResults.push_back(
              builder.create<tensor::DimOp>(op->getLoc(), res, 0));
        }
        Value sum = dim0OfResults[0];
        for (size_t i = 1; i < dim0OfResults.size(); ++i) {
          sum =
              builder.create<shape::AddOp>(op->getLoc(), sum, dim0OfResults[i]);
        }
        Value dim0OfOperand =
            builder.create<tensor::DimOp>(op->getLoc(), op->getOperand(0), 0);
        builder.create<shape_ext::MeetOp>(op->getLoc(), sum, dim0OfOperand);
        return success();
      });
}

/// See DynamicPartition's signature on
/// https://www.tensorflow.org/api_docs/python/tf/dynamic_partition
void mlir::registerDynamicPartitionInferBoundedReturnTypeComponents() {
  static InferBoundedReturnTypeComponentsRegistration shapeRegister(
      getDynamicPartitionName(),
      [](MLIRContext *context, Optional<Location>, ValueShapeRange operands,
         DictionaryAttr attr, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        auto numPartition = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                                .getAs<IntegerAttr>("num_partitions")
                                .getInt();
        if (ShapedType shapedType =
                operands[0].getType().dyn_cast_or_null<ShapedType>()) {
          inferredReturnTypes.append(numPartition, shapedType);
          return success();
        }
        return failure();
      });
}