//===- LoopUtils.h ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_LOOPUTILS_H
#define BYTEIR_UTILS_LOOPUTILS_H

#include "llvm/ADT/Optional.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Block;
class LoopLikeOpInterface;
class OpBuilder;
class Value;

namespace scf {
class ForOp;
} // namespace scf

Value getInductionVar(LoopLikeOpInterface looplike);

Value getLoopStep(LoopLikeOpInterface looplike);

// return lb + idx * step
Value createLinearIndexValue(OpBuilder &b, Value lb, Value step, int64_t idx);

// return lb (of looplike) + idx * step (of looplike)
Value createIndexValue(OpBuilder &b, LoopLikeOpInterface looplike, int64_t idx);

// return loopIV (of looplike) + idx * step
Value createRelativeIndexValue(OpBuilder &b, LoopLikeOpInterface looplike,
                               int64_t idx);

// create if index < ub (of looplike)
// and return the block of created if
Block *createGuardedBranch(OpBuilder &b, Value index,
                           LoopLikeOpInterface looplike);

// change loop step by multiplying original step by cnt
void multiplyLoopStep(OpBuilder &b, LoopLikeOpInterface looplike, int64_t cnt);
  
// Return ConstantTripCount for a ForOp
// Return None, if not applicable.
llvm::Optional<uint64_t> getConstantTripCount(scf::ForOp forOp);

LogicalResult loopUnrollFull(scf::ForOp forOp);

LogicalResult loopUnrollUpToFactor(scf::ForOp forOp, uint64_t unrollFactor);


} // namespace mlir

#endif // BYTEIR_UTILS_LOOPUTILS_H
