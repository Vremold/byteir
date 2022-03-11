//===- Passes.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_MEMREF_PASSES_H
#define BYTEIR_MEMREF_PASSES_H

#include "byteir/Dialect/MemRef/Transforms/ApplyMemRefAffineLayout.h"
#include "byteir/Dialect/MemRef/Transforms/ReifyAlloc.h"
#include "byteir/Dialect/MemRef/Transforms/SimplifyView.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Dialect/MemRef/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_MEMREF_PASSES_H
