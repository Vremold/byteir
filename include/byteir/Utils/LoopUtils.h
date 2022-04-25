//===- LoopUtils.h ---------------------------------------------*--- C++
//-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_LOOPUTILS_H
#define BYTEIR_UTILS_LOOPUTILS_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/Optional.h"

namespace mlir {
class Block;
class FuncOp;
class LoopLikeOpInterface;
class OpBuilder;
class Operation;
class Value;

namespace scf {
class ForOp;
} // namespace scf

Value getInductionVar(LoopLikeOpInterface looplike);

Value getLoopStep(LoopLikeOpInterface looplike);

// return lb + idx * step
Value createLinearIndexValue(OpBuilder &b, Value lb, Value step, Value idx);

// return lb + idx * step
Value createLinearIndexValue(OpBuilder &b, Value lb, Value step, int64_t idx);

// return lb (of looplike) + idx * step (of looplike)
Value createIndexValue(OpBuilder &b, LoopLikeOpInterface looplike, Value idx);

// return lb (of looplike) + idx * step (of looplike)
Value createIndexValue(OpBuilder &b, LoopLikeOpInterface looplike, int64_t idx);

// return loopIV (of looplike) + idx * step
Value createRelativeIndexValue(OpBuilder &b, LoopLikeOpInterface looplike,
                               int64_t idx);

// check whether 'val' >= ub (of looplike).
// return false if unknown statically
bool confirmGEUpperBound(Value val, LoopLikeOpInterface looplike);

// create if index < ub (of looplike)
// and return the block of created if
Block *createGuardedBranch(OpBuilder &b, Value index,
                           LoopLikeOpInterface looplike);

// change loop step by multiplying original step by multiplier
void multiplyLoopStep(OpBuilder &b, LoopLikeOpInterface looplike,
                      int64_t multiplier);

void multiplyLoopStep(OpBuilder &b, LoopLikeOpInterface looplike,
                      Value multiplier);

void setLoopLowerBound(OpBuilder &b, LoopLikeOpInterface looplike, Value lb);

// lb = lb + val
void addLoopLowerBound(OpBuilder &b, LoopLikeOpInterface looplike, Value val);

// Return ConstantTripCount for a looplike
// Return None, if not applicable.
llvm::Optional<uint64_t> getConstantTripCount(LoopLikeOpInterface looplike,
                                              int64_t stepMultiplier = 1);
// Return ConstantTripCount for a ForOp
// Return None, if not applicable.
llvm::Optional<uint64_t> getConstantTripCount(scf::ForOp forOp,
                                              int64_t stepMultiplier = 1);

void gatherLoopsWithDepth(FuncOp func, unsigned depth,
                          SmallVectorImpl<Operation *> &collector);

// create a scf::ForOp(0, 1, 1) if possible
// if FuncOp is trivally empty return None.
llvm::Optional<scf::ForOp> createTrivialSCFForIfHaveNone(FuncOp);

LogicalResult loopUnrollFull(scf::ForOp forOp);

LogicalResult loopUnrollUpToFactor(scf::ForOp forOp, uint64_t unrollFactor);

} // namespace mlir

#endif // BYTEIR_UTILS_LOOPUTILS_H
