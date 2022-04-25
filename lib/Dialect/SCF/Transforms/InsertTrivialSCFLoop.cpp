//===- InsertTrivialSCFLoop.cpp ------------------------------------ C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/SCF/Transforms/InsertTrivialSCFLoop.h"
#include "PassDetail.h"
#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace mlir::scf;

namespace {

struct InsertTrivialSCFLoopPass
    : public InsertTrivialSCFLoopBase<InsertTrivialSCFLoopPass> {
  InsertTrivialSCFLoopPass(llvm::StringRef anchor)
      : InsertTrivialSCFLoopBase() {
    anchorTag = anchor.str();
  }
  void runOnOperation() override {
    FuncOp funcOp = getOperation();

    // skip non-anchored
    if (!anchorTag.empty() && !funcOp->hasAttr(anchorTag)) {
      return;
    }

    (void)createTrivialSCFForIfHaveNone(funcOp);
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createInsertTrivialSCFLoopPass(llvm::StringRef anchor) {
  return std::make_unique<InsertTrivialSCFLoopPass>(anchor);
}
