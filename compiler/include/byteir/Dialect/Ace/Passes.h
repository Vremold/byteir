//===- Passes.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_ACE_PASSES_H
#define BYTEIR_ACE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

/// Creates an instance of `ace` dialect bufferization pass.
std::unique_ptr<OperationPass<func::FuncOp>> createAceBufferizePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Dialect/Ace/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_ACE_PASSES_H
