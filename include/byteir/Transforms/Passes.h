//===- Passes.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_PASSES_H
#define BYTEIR_TRANSFORMS_PASSES_H

#include "byteir/Transforms/CMAE.h"
#include "byteir/Transforms/RewriteOpToStdCall.h"
#include "byteir/Transforms/SetArgShape.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Transforms/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_PASSES_H
