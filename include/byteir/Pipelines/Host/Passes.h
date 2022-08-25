//===- Passes.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_HOST_PASSES_H
#define BYTEIR_PIPELINES_HOST_PASSES_H

#include "byteir/Pipelines/Host/HostOpt.h"
#include "byteir/Pipelines/Host/ToLLVM.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Pipelines/Host/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_PIPELINES_HOST_PASSES_H
