//===- ApplyMemRefLayout.h ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MEMREF_TRANSFORMS_REIFYALLOC_H
#define BYTEIR_DIALECT_MEMREF_TRANSFORMS_REIFYALLOC_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {

class RewritePatternSet;

// return shape and operands for an alloc-like op
void reifyAllocLikeShapeAndOperands(
  ArrayRef<int64_t> oldShape,
  ValueRange oldOperands,
  SmallVectorImpl<int64_t>& newShape,
  SmallVectorImpl<Value>& newOperands);

void populateReifyAllocLikePatterns(RewritePatternSet& patterns);

std::unique_ptr<OperationPass<FuncOp>> createReifyAllocPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MEMREF_TRANSFORMS_REIFYALLOC_H