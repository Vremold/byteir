//===- Passes.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_MHLO_PASSES_H
#define BYTEIR_MHLO_PASSES_H

#include "byteir/Dialect/mhlo/transforms/ArithOptimize.h"
#include "byteir/Dialect/mhlo/transforms/ConvBiasActFusion.h"
#include "byteir/Dialect/mhlo/transforms/DotTransposeFusion.h"
#include "byteir/Dialect/mhlo/transforms/ElementFusion.h"
#include "byteir/Dialect/mhlo/transforms/FusionOutlining.h"
#include "byteir/Dialect/mhlo/transforms/HloFolder.h"
#include "byteir/Dialect/mhlo/transforms/HloTransposeDotToDotGeneral.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Dialect/mhlo/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_MHLO_PASSES_H
