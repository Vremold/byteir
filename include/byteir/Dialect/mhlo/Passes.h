//===- Passes.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_MHLO_PASSES_H
#define BYTEIR_MHLO_PASSES_H

#include "byteir/Dialect/mhlo/Transforms/ConvBiasActFusion.h"
#include "byteir/Dialect/mhlo/Transforms/DotTransposeFusion.h"
#include "byteir/Dialect/mhlo/Transforms/ElementFusion.h"
#include "byteir/Dialect/mhlo/Transforms/FusionOutlining.h"
#include "byteir/Dialect/mhlo/Transforms/HloFolder.h"
#include "byteir/Dialect/mhlo/Transforms/HloTransposeDotToDotGeneral.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Dialect/mhlo/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_MHLO_PASSES_H
