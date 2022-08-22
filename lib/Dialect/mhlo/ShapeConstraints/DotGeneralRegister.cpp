//===- DotGeneralRegister.cpp ---------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/ShapeConstraints/Register.h"

#include "byteir/Dialect/Shape/ShapeExtOps.h"
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
        auto dotGeneral = cast<mhlo::DotGeneralOp>(op);
        auto dimNumbers = dotGeneral.dot_dimension_numbers();

        // batching dimensions match
        auto lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
        auto rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
        for (int i = 0; i < lhsBatchingDims.size(); ++i) {
          auto lDim = lhsBatchingDims[i];
          auto rDim = rhsBatchingDims[i];
          Value lhsD = builder.create<tensor::DimOp>(op->getLoc(),
                                                     dotGeneral.lhs(), lDim);
          Value rhsD = builder.create<tensor::DimOp>(op->getLoc(),
                                                     dotGeneral.rhs(), rDim);
          builder.create<shape_ext::MeetOp>(op->getLoc(), lhsD, rhsD);
        }

        // contracting dimensions match
        auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
        auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
        for (int i = 0; i < lhsContractingDims.size(); ++i) {
          auto lDim = lhsContractingDims[i];
          auto rDim = rhsContractingDims[i];
          Value lhsD = builder.create<tensor::DimOp>(op->getLoc(),
                                                     dotGeneral.lhs(), lDim);
          Value rhsD = builder.create<tensor::DimOp>(op->getLoc(),
                                                     dotGeneral.rhs(), rDim);
          builder.create<shape_ext::MeetOp>(op->getLoc(), lhsD, rhsD);
        }
        return success();
      });
}
