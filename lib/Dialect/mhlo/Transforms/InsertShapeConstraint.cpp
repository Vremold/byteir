//===- InsertShapeConstraint.cpp ------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/InsertShapeConstraint.h"
#include "./PassDetail.h"
#include "byteir/Dialect/Shape/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/ShapeConstraints/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include <vector>

using namespace mlir;

namespace {

struct InsertShapeConstraintPass
    : public InsertShapeConstraintBase<InsertShapeConstraintPass> {
  InsertShapeConstraintPass()
      : InsertShapeConstraintBase<
            InsertShapeConstraintPass>::InsertShapeConstraintBase() {
    registerAllMhloShapeConstraints();
  }

  void runOnOperation() override {
    std::vector<Operation *> ops;
    FuncOp funcOp = getOperation();
    funcOp.walk([&](Operation *op) { ops.push_back(op); });

    OpBuilder builder(funcOp->getContext());

    // std::reverse(ops.begin(), ops.end());
    for (Operation *op : ops) {
      llvm::StringRef opName;

      if (auto customCall = llvm::dyn_cast<mhlo::CustomCallOp>(op)) {
        opName = customCall.call_target_name();
      } else {
        opName = op->getName().getStringRef();
      }

      if (auto insertShapeConstraintFunc = insertShapeConstraint(opName)) {
        LogicalResult status = insertShapeConstraintFunc(op, builder);
        (void)status; // Suppress unused warning
      }
    }
  };
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createInsertShapeConstraintPass() {
  return std::make_unique<InsertShapeConstraintPass>();
}
