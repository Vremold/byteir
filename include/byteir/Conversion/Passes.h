//===- Passes.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_PASSES_H
#define BYTEIR_CONVERSION_PASSES_H

#include "byteir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "byteir/Conversion/HloToLHlo/HloToLHlo.h"
#include "byteir/Conversion/HloToLinalg/HloToLinalg.h"
#include "byteir/Conversion/LmhloToMemref/LmhloToMemref.h"
#include "byteir/Conversion/ToAce/MhloToAce.h"
#include "byteir/Conversion/ToAce/UnregisteredToAce.h"
#include "byteir/Conversion/ToByre/ToByre.h"
#include "byteir/Conversion/ToGPU/ToGPU.h"
#include "byteir/Conversion/ToHlo/ArithToMhlo.h"
#include "byteir/Conversion/ToPTX/ToPTX.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_CONVERSION_PASSES_H
