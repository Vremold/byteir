//===- DynamicPartitionRegister.h -----------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/ReifyShapes/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

void mlir::registerDynamicStitchReifyReturnTypeShapes() {
  static ReifyReturnTypeShapesRegistration shapeRegister(
      getDynamicStitchName(),
      [](Operation *op, OpBuilder &builder, ValueRange operands,
         SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) {
        unsigned numOperands = op->getNumOperands();
        unsigned halfNum = numOperands / 2;
        SmallVector<Value> data;
        data.reserve(halfNum);
        for (unsigned i = 0; i < halfNum; ++i) {
          data.push_back(op->getOperand(i));
        }

        bool allRankedTensor = llvm::all_of(
            data, [](Value v) { return v.getType().isa<RankedTensorType>(); });
        if (!allRankedTensor)
          return failure();

        Value dim0 = builder.create<tensor::DimOp>(op->getLoc(), data[0], 0);
        for (unsigned i = 1; i < halfNum; ++i) {
          Value ithDim0 =
              builder.create<tensor::DimOp>(op->getLoc(), data[i], 0).result();
          dim0 = builder.create<shape::AddOp>(op->getLoc(), dim0, ithDim0);
        }
        SmallVector<Value> dims;
        dims.push_back(dim0);
        for (int64_t i = 1;
             i < data[0].getType().cast<RankedTensorType>().getRank(); ++i) {
          dims.push_back(
              builder.create<tensor::DimOp>(op->getLoc(), data[0], i));
        }
        reifiedReturnShapes.push_back(
            builder.create<tensor::FromElementsOp>(op->getLoc(), dims));

        return success();
      });
}