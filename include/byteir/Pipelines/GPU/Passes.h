//===- Passes.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_GPU_PASSES_H
#define BYTEIR_PIPELINES_GPU_PASSES_H

#include "byteir/Pipelines/GPU/GPUOpt.h"
#include "byteir/Pipelines/GPU/LinalgMemrefGPU.h"
#include "byteir/Pipelines/GPU/NVVMCodegen.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Pipelines/GPU/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_PIPELINES_GPU_PASSES_H
