//===- Passes.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_AFFINE_PASSES_H
#define BYTEIR_AFFINE_PASSES_H

#include "byteir/Dialect/Affine/Transforms/AffineLoopFusionEx.h"
#include "byteir/Dialect/Affine/Transforms/RewriteAffineToMemref.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Dialect/Affine/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_AFFINE_PASSES_H
