//===- LoopUtils.cpp
//-------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;

Value mlir::getInductionVar(LoopLikeOpInterface looplike) {
  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    return forOp.getInductionVar();
  }
  return Value();
}

Value mlir::getLoopStep(LoopLikeOpInterface looplike) {
  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    return forOp.getStep();
  }
  return Value();
}

// return lbs + idx * step
Value mlir::createLinearIndexValue(OpBuilder &b, Value lb, Value step,
                                   int64_t idx) {
  auto loc = lb.getLoc();
  Value cntValue = b.create<ConstantIndexOp>(loc, idx);
  auto mul = b.create<MulIOp>(loc, cntValue, step);
  auto add = b.create<AddIOp>(loc, lb, mul);
  return add.getResult();
}

// return lbs + idx * step
Value mlir::createIndexValue(OpBuilder &b, LoopLikeOpInterface looplike,
                             int64_t idx) {

  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    auto lb = forOp.getLowerBound();
    auto step = forOp.getStep();
    return createLinearIndexValue(b, lb, step, idx);
  }
  return Value();
}

// return loopIV + idx * step
Value mlir::createRelativeIndexValue(OpBuilder &b, LoopLikeOpInterface looplike,
                                     int64_t idx) {

  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    auto loopIV = getInductionVar(looplike);
    auto step = forOp.getStep();
    return createLinearIndexValue(b, loopIV, step, idx);
  }
  return Value();
}

// create if index < ub (of looplike)
// and return the block of created if
Block *mlir::createGuardedBranch(OpBuilder &b, Value index,
                                 LoopLikeOpInterface looplike) {
  auto loc = looplike.getLoc();

  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    auto ub = forOp.getUpperBound();
    Value cond = b.create<CmpIOp>(loc, CmpIPredicate::slt, index, ub);
    auto ifOp = b.create<scf::IfOp>(loc, cond, /*withElseRegion*/ false);
    return ifOp.getBody(0);
  }
  return nullptr;
}

// change loop step by multiplying original step by cnt
void mlir::multiplyLoopStep(OpBuilder &b, LoopLikeOpInterface looplike,
                            int64_t cnt) {
  b.setInsertionPoint(looplike);
  auto loc = looplike.getLoc();
  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    auto step = forOp.getStep();
    Value cntValue = b.create<ConstantIndexOp>(loc, cnt);
    auto mul = b.create<MulIOp>(loc, cntValue, step);
    forOp.setStep(mul.getResult());
  }
}

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

    if (tripCnt >= 0)
      return tripCnt;
  }
  return llvm::None;
}

LogicalResult mlir::loopUnrollFull(scf::ForOp forOp) {
  auto mayBeConstantCount = getConstantTripCount(forOp);
  if (!mayBeConstantCount.hasValue())
    return failure();
  return loopUnrollByFactor(forOp, mayBeConstantCount.getValue());
}

LogicalResult mlir::loopUnrollUpToFactor(scf::ForOp forOp,
                                         uint64_t unrollFactor) {
  auto mayBeConstantCount = getConstantTripCount(forOp);
  if (mayBeConstantCount.hasValue() &&
      mayBeConstantCount.getValue() <= unrollFactor) {
    return loopUnrollByFactor(forOp, mayBeConstantCount.getValue());
  }
  return loopUnrollByFactor(forOp, unrollFactor);
}
