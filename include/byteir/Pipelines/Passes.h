//===- Passes.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_PASSES_H
#define BYTEIR_PIPELINES_PASSES_H

#include "byteir/Pipelines/AffineOpt.h"
#include "byteir/Pipelines/AllOpt.h"
#include "byteir/Pipelines/ByreHost.h"
#include "byteir/Pipelines/ByreOpt.h"
#include "byteir/Pipelines/HloOpt.h"
#include "byteir/Pipelines/LinalgTensorOpt.h"
#include "byteir/Pipelines/SCFOpt.h"
#include "byteir/Pipelines/TotalBufferize.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Pipelines/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_PIPELINES_PASSES_H
