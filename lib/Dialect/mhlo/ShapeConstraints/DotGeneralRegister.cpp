//===- DotGeneralRegister.cpp ---------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Shape/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/ShapeConstraints/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
void mlir::registerDotGeneralShapeConstraints() {
  static InsertShapeConstraintRegistration shapeRegister(
      mhlo::DotGeneralOp::getOperationName(),
      [](Operation *op, OpBuilder &builder) {
        builder.setInsertionPointAfter(op);
        auto dot_general = cast<mhlo::DotGeneralOp>(op);
        auto dim_numbers = dot_general.dot_dimension_numbers();

        // batching dimensions match
        auto lhs_batching_dims = dim_numbers.getLhsBatchingDimensions();
        auto rhs_batching_dims = dim_numbers.getRhsBatchingDimensions();
        for (int i = 0; i < lhs_batching_dims.size(); ++i) {
          auto lDim = lhs_batching_dims[i];
          auto rDim = rhs_batching_dims[i];
          Value lhs_d = builder.create<tensor::DimOp>(op->getLoc(),
                                                      dot_general.lhs(), lDim);
          Value rhs_d = builder.create<tensor::DimOp>(op->getLoc(),
                                                      dot_general.rhs(), rDim);
          builder.create<shape_ext::MeetOp>(op->getLoc(), lhs_d, rhs_d);
        }

        // contracting dimensions match
        auto lhs_contracting_dims = dim_numbers.getLhsContractingDimensions();
        auto rhs_contracting_dims = dim_numbers.getRhsContractingDimensions();
        for (int i = 0; i < lhs_contracting_dims.size(); ++i) {
          auto lDim = lhs_contracting_dims[i];
          auto rDim = rhs_contracting_dims[i];
          Value lhs_d = builder.create<tensor::DimOp>(op->getLoc(),
                                                      dot_general.lhs(), lDim);
          Value rhs_d = builder.create<tensor::DimOp>(op->getLoc(),
                                                      dot_general.rhs(), rDim);
          builder.create<shape_ext::MeetOp>(op->getLoc(), lhs_d, rhs_d);
        }
        return success();
      });
}
