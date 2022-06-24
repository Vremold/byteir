//===- InsertShapeConstraint.cpp ------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/InsertShapeConstraint.h"
#include "./PassDetail.h"
#include "byteir/Dialect/Shape/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include <vector>

using namespace mlir;

namespace {

void handleDynamicPartition(Operation *op) {
  SmallVector<Value> dim0OfResults;
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  for (Value res : op->getResults()) {
    dim0OfResults.push_back(
        builder.create<tensor::DimOp>(op->getLoc(), res, 0));
  }
  Value sum = dim0OfResults[0];
  for (size_t i = 1; i < dim0OfResults.size(); ++i) {
    sum = builder.create<shape::AddOp>(op->getLoc(), sum, dim0OfResults[i]);
  }
  Value dim0OfOperand =
      builder.create<tensor::DimOp>(op->getLoc(), op->getOperand(0), 0);
  builder.create<shape_ext::MeetOp>(op->getLoc(), sum, dim0OfOperand);
}

struct InsertShapeConstraintPass
    : public InsertShapeConstraintBase<InsertShapeConstraintPass> {

  void runOnOperation() override {
    std::vector<Operation *> ops;
    FuncOp funcOp = getOperation();
    funcOp.walk([&](Operation *op) { ops.push_back(op); });

    for (Operation *op : ops) {
      if (auto customCall = llvm::dyn_cast<mhlo::CustomCallOp>(op)) {
        if (customCall.call_target_name() == getDynamicPartitionName()) {
          handleDynamicPartition(op);
        }
      }
    }
  };
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createInsertShapeConstraintPass() {
  return std::make_unique<InsertShapeConstraintPass>();
}
