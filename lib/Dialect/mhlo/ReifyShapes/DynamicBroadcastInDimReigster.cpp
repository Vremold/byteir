//===- DynamicBroadcastInDimRegister.h -------------------------*--- C++-*-===//
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
