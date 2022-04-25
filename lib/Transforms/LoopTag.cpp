//===- FuncTag.cpp ------------------------------------------------- C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/LoopTag.h"
#include "./PassDetail.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include <iostream>

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

    auto ctx = funcOp.getContext();
    parseConcatAttr(attachAttr, attrName, attrType, attrValue);

    for (auto *op : collector) {
      if (op->getName().getStringRef() != loopType) {
        continue;
      }

      if (attrType == "Unit") {
        op->setAttr(attrName, UnitAttr::get(ctx));
      } else if (attrType == "String") {
        op->setAttr(attrName, StringAttr::get(ctx, attrValue));
      } else if (attrType == "I32") {
        int intVal = std::stoi(attrValue);
        op->setAttr(attrName,
                    IntegerAttr::get(IntegerType::get(ctx, 32), intVal));
      } else if (attrType == "F32") {
        float f32Val = std::stof(attrValue);
        op->setAttr(attrName, FloatAttr::get(Float32Type::get(ctx), f32Val));
      } else {
        op->emitOpError() << "unsupport attachAttr";
      }
    }
  }

  std::string attrName;
  std::string attrType;
  std::string attrValue;
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLoopTagPass(llvm::StringRef anchorTag, const std::string &attachTag,
                        unsigned depth, const std::string &loopType) {
  return std::make_unique<LoopTagPass>(anchorTag.str(), attachTag, depth,
                                       loopType);
}
