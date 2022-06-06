//===- Passes.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_MHLO_PASSES_H
#define BYTEIR_MHLO_PASSES_H

#include "byteir/Dialect/mhlo/Transforms/BoundedShapeInference.h"
#include "byteir/Dialect/mhlo/Transforms/ClusterConstraint.h"
#include "byteir/Dialect/mhlo/Transforms/ConvBackwardFusion.h"
#include "byteir/Dialect/mhlo/Transforms/ConvBiasActFusion.h"
#include "byteir/Dialect/mhlo/Transforms/DotTransposeFusion.h"
#include "byteir/Dialect/mhlo/Transforms/FusionOutlining.h"
#include "byteir/Dialect/mhlo/Transforms/GenericFusion.h"
#include "byteir/Dialect/mhlo/Transforms/HloFolder.h"
#include "byteir/Dialect/mhlo/Transforms/HloMove.h"
#include "byteir/Dialect/mhlo/Transforms/HloTransposeDotToDotGeneral.h"
#include "byteir/Dialect/mhlo/Transforms/IOConvertFusion.h"
#include "byteir/Dialect/mhlo/Transforms/LayoutTransformation.h"
#include "byteir/Dialect/mhlo/Transforms/ReduceFusion.h"
#include "byteir/Dialect/mhlo/Transforms/RewriteWithConstraint.h"
#include "byteir/Dialect/mhlo/Transforms/ShapeReification.h"
#include "byteir/Dialect/mhlo/Transforms/StaticShapeInference.h"
#include "byteir/Dialect/mhlo/Transforms/TrivialFusion.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Dialect/mhlo/Passes.h.inc"

// also all pass including ones from td and non-td
inline void registerByteIRMhloPassesExt() {
  // ones from td
  registerByteIRMhloPasses();

  // ones not from td
  // register createElementFusionPass
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createElementFusionPass();
  });

  // register createMatmulEpilogueFusionPass
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createMatmulEpilogueFusionPass();
  });
}

} // namespace mlir

#endif // BYTEIR_MHLO_PASSES_H
