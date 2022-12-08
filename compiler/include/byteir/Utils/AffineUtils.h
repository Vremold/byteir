//===- AffineUtils.h ----------------------------------------------- C++---===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_AFFINEUTILS_H
#define BYTEIR_UTILS_AFFINEUTILS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

/// find iteration index through dim and inversePermutation
/// E.g. if affineMap = (d0, d1, d2)-> (d0, d2), dim = 1
/// return 2  (from d2)
FailureOr<unsigned> getIterAxisFromDim(AffineMap affineMap, unsigned dimIndex);

} // namespace mlir

#endif // BYTEIR_UTILS_AFFINEUTILS_H
