//===- Passes.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_BYRE_PASSES_H
#define BYTEIR_BYRE_PASSES_H

#include "byteir/Dialect/Byre/transforms/Fold.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Dialect/Byre/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_BYRE_PASSES_H
