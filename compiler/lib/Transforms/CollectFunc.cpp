//===- CollectFunc.cpp ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/CollectFunc.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct CollectFuncPass : public CollectFuncBase<CollectFuncPass> {
  CollectFuncPass(llvm::StringRef tag) : CollectFuncBase() {
    this->anchorAttr = tag.str();
  }

  void runOnOperation() override {
    if (anchorAttr.empty())
      return;

    auto m = getOperation();

    SmallVector<Operation *> removeOps;
    for (auto &op : m.getBody()->without_terminator()) {
      if (!isa<func::FuncOp>(op)) {
        removeOps.push_back(&op);
      }
    }

    // funcOp not in m.getBody()->without_terminator()
    for (auto funcOp : m.getOps<func::FuncOp>()) {
      // only consider public
      if (funcOp.isPublic() && !funcOp->hasAttr(anchorAttr)) {
        removeOps.push_back(funcOp);
      }
    }

    for (auto op : removeOps) {
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createCollectFuncPass(llvm::StringRef anchorTag) {
  return std::make_unique<CollectFuncPass>(anchorTag);
}
