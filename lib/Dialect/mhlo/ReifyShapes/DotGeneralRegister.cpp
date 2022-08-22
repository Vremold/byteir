//===- DotGeneralRegister.cpp ---------------------------------*--- C++ -*-===//
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

void mlir::registerDotGeneralReifyReturnTypeShapes() {
  static ReifyReturnTypeShapesRegistration shapeRegister(
      mhlo::DotGeneralOp::getOperationName(),
      [](Operation *op, OpBuilder &builder, ValueRange operands,
         SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) {
        // TODO: replace this impl after updating mlir-hlo
        // for now this is copied impl from
        // mlir-hlo/lib/Dialect/mhlo/IR/hlo_ops.cc
        auto dotGeneral = dyn_cast<mhlo::DotGeneralOp>(op);
        auto lhsType = dotGeneral.lhs().getType().dyn_cast<ShapedType>();
        auto rhsType = dotGeneral.rhs().getType().dyn_cast<ShapedType>();
        if (!lhsType || !rhsType) {
          return failure();
        }

        mhlo::DotGeneralOp::Adaptor adaptor(operands);
        auto dimNumbers = dotGeneral.dot_dimension_numbers();
        SmallVector<Value> dimensions;
        for (const int64_t lhsDim : dimNumbers.getLhsBatchingDimensions()) {
          dimensions.push_back(builder.create<tensor::DimOp>(
              dotGeneral.getLoc(), adaptor.lhs(), lhsDim));
        }

        for (int64_t i = 0; i < lhsType.getRank(); i++) {
          if (!llvm::is_contained(dimNumbers.getLhsContractingDimensions(),
                                  i) &&
              !llvm::is_contained(dimNumbers.getLhsBatchingDimensions(), i)) {
            dimensions.push_back(builder.create<tensor::DimOp>(
                dotGeneral.getLoc(), adaptor.lhs(), i));
          }
        }
        for (int64_t i = 0; i < rhsType.getRank(); i++) {
          if (!llvm::is_contained(dimNumbers.getRhsContractingDimensions(),
                                  i) &&
              !llvm::is_contained(dimNumbers.getRhsBatchingDimensions(), i)) {
            dimensions.push_back(builder.create<tensor::DimOp>(
                dotGeneral.getLoc(), adaptor.rhs(), i));
          }
        }

        reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
            dotGeneral.getLoc(), dimensions));
        return success();
      });
}
