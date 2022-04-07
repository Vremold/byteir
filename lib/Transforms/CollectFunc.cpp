//===- CollectFunc.cpp -----------------------------------------*--- C++
//-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/CollectFunc.h"
#include "./PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"

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
      if (!isa<FuncOp>(op)) {
        removeOps.push_back(&op);
      }
    }

    // funcOp not in m.getBody()->without_terminator()
    for (auto funcOp : m.getOps<FuncOp>()) {
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
