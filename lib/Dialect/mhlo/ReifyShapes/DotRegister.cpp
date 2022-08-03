//===- DotRegister.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/ReifyShapes/Register.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

void mlir::registerDotReifyReturnTypeShapes() {
  static ReifyReturnTypeShapesRegistration shapeRegister(
      mhlo::DotOp::getOperationName(),
      [](Operation *op, OpBuilder &builder, ValueRange operands,
         SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) {
        auto dotOp = cast<mhlo::DotOp>(op);
        auto lhs_type = dotOp.lhs().getType().dyn_cast<ShapedType>();
        auto rhs_type = dotOp.rhs().getType().dyn_cast<ShapedType>();
        if (!lhs_type || !rhs_type || !lhs_type.hasRank() ||
            !rhs_type.hasRank()) {
          return failure();
        }

        mhlo::DotOp::Adaptor adaptor(operands);
        auto lhs = adaptor.lhs();
        auto rhs = adaptor.rhs();
        SmallVector<Value> dimensions;

        // vector dot vector
        if (1 == lhs_type.getRank() && 1 == rhs_type.getRank()) {
          return success();
        }
        // matrix dot vector
        else if (2 == lhs_type.getRank() && 1 == rhs_type.getRank()) {
          dimensions.push_back(
              builder.create<tensor::DimOp>(dotOp.getLoc(), lhs, 0));
        }
        // vector dot matrix
        else if (1 == lhs_type.getRank() && 2 == rhs_type.getRank()) {
          dimensions.push_back(
              builder.create<tensor::DimOp>(dotOp.getLoc(), rhs, 1));
        }
        // matrix dot matrix
        else if (2 == lhs_type.getRank() && 2 == rhs_type.getRank()) {
          dimensions.push_back(
              builder.create<tensor::DimOp>(dotOp.getLoc(), lhs, 0));
          dimensions.push_back(
              builder.create<tensor::DimOp>(dotOp.getLoc(), rhs, 1));
        } else {
          return failure();
        }
        reifiedReturnShapes.push_back(
            builder.create<tensor::FromElementsOp>(dotOp.getLoc(), dimensions));
        return success();
      });
}