//===- RemoveFuncBody.cpp -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/RemoveFuncBody.h"
#include "./PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"

using namespace mlir;

namespace {

struct RemoveFuncBodyPass : public RemoveFuncBodyBase<RemoveFuncBodyPass> {

  RemoveFuncBodyPass(const std::string &anchor, bool disableForcePrivate)
      : RemoveFuncBodyBase() {
    this->anchorAttr = anchor;
    this->disableForcePrivate = disableForcePrivate;
  }

  void runOnOperation() override {
    // early terminate if empty anchor
    if (anchorAttr.empty()) {
      return;
    }

    auto f = getOperation();

    // early terminate if func has no anchor or func is already empty
    if (!f->hasAttr(anchorAttr) || f.empty()) {
      return;
    }

    if (f.isPublic()) {
      // early terminate if func is public and disableForcePrivate is true
      if (disableForcePrivate) {
        return;
      }
      // convert public to private
      f.setPrivate();
    }

    // remove body
    f.getBody().getBlocks().clear();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createRemoveFuncBodyPass(llvm::StringRef anchorTag,
                               bool disableForcePrivate) {
  return std::make_unique<RemoveFuncBodyPass>(anchorTag.str(),
                                              disableForcePrivate);
}
