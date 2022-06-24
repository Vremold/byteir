//===- Passes.h ---------------------------------------------------- C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_SHAPE_PASSES_H
#define BYTEIR_SHAPE_PASSES_H

#include "byteir/Dialect/Shape/Transforms/InsertTieShape.h"
#include "byteir/Dialect/Shape/Transforms/ResolveShapeConstraint.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Dialect/Shape/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_SHAPE_PASSES_H
