//===- LinalgPrefetch.h --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_LINALGPREFETCH_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_LINALGPREFETCH_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {

// TODO add a namespace if conflict
constexpr StringRef getPrefetchAttrName() { return "__byteir_prefetch__"; }

std::unique_ptr<OperationPass<FuncOp>>
createLinalgPrefetchPass(int64_t prefetchCnt = 1, bool unroll = false);

} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_LINALGPREFETCH_H