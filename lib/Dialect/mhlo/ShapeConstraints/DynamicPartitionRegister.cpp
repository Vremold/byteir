//===- DynamicPartitionRegister.cpp ---------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Shape/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/ShapeConstraints/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/Sequence.h"

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
