//===- IRRewrite.h --------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_IRREWRITE_H
#define BYTEIR_UTILS_IRREWRITE_H

#include <functional>

namespace mlir {
class Operation;
class Block;

// replicate specific ops satisfying func
void ReplicateDefiningOp(Block* block, std::function<bool(Operation*)> checkFunc);

} // namespace mlir

#endif // BYTEIR_UTILS_IRREWRITE_H