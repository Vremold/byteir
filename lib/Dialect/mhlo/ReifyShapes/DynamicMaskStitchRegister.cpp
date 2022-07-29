//===- DynamicMaskPartitionRegister.h -----------------------------*--- C++
//-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/ReifyShapes/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

void mlir::registerDynamicMaskStitchReifyReturnTypeShapes() {
  static ReifyReturnTypeShapesRegistration shapeRegister(
      getDynamicMaskStitchName(),
      [](Operation *op, OpBuilder &builder, ValueRange operands,
         SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) {
        unsigned numOperands = op->getNumOperands();

        Value dim0 = builder.create<tensor::DimOp>(
            op->getLoc(), op->getOperand(numOperands - 1), 0);
        SmallVector<Value> dims;
        dims.push_back(dim0);
        for (int64_t i = 1;
             i < op->getOperand(0).getType().cast<RankedTensorType>().getRank();
             ++i) {
          dims.push_back(builder.create<tensor::DimOp>(op->getLoc(),
                                                       op->getOperand(0), i));
        }
        reifiedReturnShapes.push_back(
            builder.create<tensor::FromElementsOp>(op->getLoc(), dims));

        return success();
      });
}