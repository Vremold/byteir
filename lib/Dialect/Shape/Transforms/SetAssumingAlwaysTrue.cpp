//===- SetAssumingAlwaysTrue.cpp ----------------------------------- C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Shape/Transforms/SetAssumingAlwaysTrue.h"
#include "PassDetail.h"
#include "byteir/Dialect/Shape/ShapeExtOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace {

struct SetAssumingAlwaysTruePass
    : public SetAssumingAlwaysTrueBase<SetAssumingAlwaysTruePass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(&funcOp.front().front());

    auto witnessTrueValue = builder.create<shape::ConstWitnessOp>(
        UnknownLoc::get(&getContext()), builder.getBoolAttr(true));
    funcOp.walk([&](Operation *op) {
      if (auto assumingOp = dyn_cast<shape::AssumingOp>(op)) {
        auto produceOp = assumingOp.getWitness().getDefiningOp();
        produceOp->replaceAllUsesWith(witnessTrueValue);
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createSetAssumingAlwaysTruePass() {
  return std::make_unique<SetAssumingAlwaysTruePass>();
}
