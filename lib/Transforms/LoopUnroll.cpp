//===- LoopUnroll.cpp --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/LoopUnroll.h"
#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

#include "./PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::memref;

namespace {

// some code is from Mlir's TestLoopUnrolling
static unsigned getNestingDepth(Operation *op) {
  Operation *currOp = op;
  unsigned depth = 0;
  while ((currOp = currOp->getParentOp())) {
    if (isa<LoopLikeOpInterface>(currOp))
      depth++;
  }
  return depth;
}

void collectCandidateLoops(func::FuncOp func,
                           SmallVectorImpl<LoopLikeOpInterface> &loops,
                           int depth) {

  auto ctx = func.getContext();
  // collect depth
  if (depth >= 0) {
    unsigned unsigned_depth = static_cast<unsigned>(depth);
    func.walk([&](LoopLikeOpInterface loop) {
      if (getNestingDepth(loop) == unsigned_depth &&
          !loop->hasAttr(getByteIRUnorllAttrName())) {
        // if not anchored, anchor it
        loop->setAttr(getByteIRUnorllAttrName(), UnitAttr::get(ctx));
      }
    });
  }

  // collect all anchored
  func.walk([&](LoopLikeOpInterface loop) {
    if (loop->hasAttr(getByteIRUnorllAttrName())) {
      loops.push_back(loop);
      // remove attr after collecting
      loop->removeAttr(getByteIRUnorllAttrName());
    }
  });
}

void unrollLoop(LoopLikeOpInterface loop, unsigned unrollFactor,
                bool unrollUpToFactor, bool unrollFull) {
  if (auto *forOp = dyn_cast<scf::ForOp>(&loop)) {
    if (unrollUpToFactor) {
      (void)loopUnrollUpToFactor(*forOp, unrollFactor);
    } else if (unrollFull) {
      (void)loopUnrollFull(*forOp);
    } else {
      (void)loopUnrollByFactor(*forOp, unrollFactor);
    }
  } else if (auto *forOp = dyn_cast<AffineForOp>(&loop)) {
    if (unrollUpToFactor) {
      (void)loopUnrollUpToFactor(*forOp, unrollFactor);
    } else if (unrollFull) {
      (void)loopUnrollFull(*forOp);
    } else {
      (void)loopUnrollByFactor(*forOp, unrollFactor);
    }
  }
}

struct LoopUnrollPass : public LoopUnrollBase<LoopUnrollPass> {
  LoopUnrollPass(unsigned factor, bool upTo, bool full, int depth)
      : LoopUnrollBase() {
    this->unrollFactor = factor;
    this->unrollUpToFactor = upTo;
    this->unrollFull = full;
    this->depth = depth;
  }

  void runOnOperation() override {
    if (unrollFactor < 2)
      return;

    func::FuncOp func = getOperation();
    SmallVector<LoopLikeOpInterface, 4> loops;

    collectCandidateLoops(func, loops, depth);

    for (auto loop : loops) {
      unrollLoop(loop, unrollFactor, unrollUpToFactor, unrollFull);
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createByteIRLoopUnrollPass(unsigned factor, bool upTo, bool full,
                                 int depth) {
  return std::make_unique<LoopUnrollPass>(factor, upTo, full, depth);
}
