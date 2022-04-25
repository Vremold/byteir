//===- Passes.h --------------------------------------------------- C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_SCF_PASSES_H
#define BYTEIR_SCF_PASSES_H

#include "byteir/Dialect/SCF/Transforms/InsertTrivialSCFLoop.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Dialect/SCF/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_SCF_PASSES_H
