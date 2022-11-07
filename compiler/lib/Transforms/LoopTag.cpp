//===- FuncTag.cpp ------------------------------------------------- C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/LoopTag.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct LoopTagPass : public LoopTagBase<LoopTagPass> {
  LoopTagPass(const std::string &anchor, const std::string &attach,
              unsigned depth, const std::string &loopType)
      : LoopTagBase<LoopTagPass>() {
    this->anchorAttr = anchor;
    this->attachAttr = attach;
    this->depth = depth;
    this->loopType = loopType;
  }

  void runOnOperation() override {
    if (anchorAttr.empty()) {
      return;
    }

    auto funcOp = getOperation();

    if (!funcOp->hasAttr(anchorAttr)) {
      return;
    }

    SmallVector<Operation *> collector;
    gatherLoopsWithDepth(funcOp, depth, collector);

    // early termination if no gathered loops
    if (collector.empty()) {
      return;
    }

    parseConcatAttr(attachAttr, attrName, attrType, attrValue);

    for (auto *op : collector) {
      if (op->getName().getStringRef() != loopType) {
        continue;
      }

      setParsedConcatAttr(op, attrName, attrType, attrValue);
    }
  }

  std::string attrName;
  std::string attrType;
  std::string attrValue;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLoopTagPass(llvm::StringRef anchorTag, const std::string &attachTag,
                        unsigned depth, const std::string &loopType) {
  return std::make_unique<LoopTagPass>(anchorTag.str(), attachTag, depth,
                                       loopType);
}
