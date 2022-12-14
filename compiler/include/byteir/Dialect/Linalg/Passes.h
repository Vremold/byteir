//===- Passes.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_LINALG_PASSES_H
#define BYTEIR_LINALG_PASSES_H

#include "byteir/Dialect/Linalg/Transforms/FuseElementwise.h"
#include "byteir/Dialect/Linalg/Transforms/LinalgDataPlace.h"
#include "byteir/Dialect/Linalg/Transforms/LinalgPrefetch.h"
#include "byteir/Dialect/Linalg/Transforms/Tiling.h"
#include "byteir/Dialect/Linalg/Transforms/TransformInsertion.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Dialect/Linalg/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_LINALG_PASSES_H
