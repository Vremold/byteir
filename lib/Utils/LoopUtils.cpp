//===- LoopUtils.cpp -------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace llvm;
using namespace mlir;

Optional<uint64_t> mlir::getConstantTripCount(scf::ForOp forOp) {
  auto lbCstOp = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ubCstOp = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();

  if (lbCstOp && ubCstOp && stepCstOp) {
    // Constant loop bounds computation.
    int64_t lbCst = lbCstOp.value();
    int64_t ubCst = ubCstOp.value();
    int64_t stepCst = stepCstOp.value();

    // TODO: please check whether negative also works
    int64_t tripCnt = (ubCst - lbCst + stepCst - 1) / stepCst;

    if (tripCnt >= 0) return tripCnt;
  }
  return llvm::None;
}

LogicalResult mlir::loopUnrollFull(scf::ForOp forOp) {
  auto mayBeConstantCount = getConstantTripCount(forOp);
  if (!mayBeConstantCount.hasValue()) return failure();
  return loopUnrollByFactor(forOp, mayBeConstantCount.getValue());
}

LogicalResult 
mlir::loopUnrollUpToFactor(scf::ForOp forOp, uint64_t unrollFactor) {
  auto mayBeConstantCount = getConstantTripCount(forOp);
  if (mayBeConstantCount.hasValue() &&
      mayBeConstantCount.getValue() <= unrollFactor) {
    return loopUnrollByFactor(forOp, mayBeConstantCount.getValue());
  }
  return loopUnrollByFactor(forOp, unrollFactor);
}
