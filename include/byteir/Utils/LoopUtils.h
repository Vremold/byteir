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
namespace scf {
  class ForOp;
} // namespace scf

// Return ConstantTripCount for a ForOp
// Return None, if not applicable.
llvm::Optional<uint64_t> getConstantTripCount(scf::ForOp forOp);

LogicalResult loopUnrollFull(scf::ForOp forOp);

LogicalResult loopUnrollUpToFactor(scf::ForOp forOp, uint64_t unrollFactor);

} // namespace mlir

#endif // BYTEIR_UTILS_LOOPUTILS_H
