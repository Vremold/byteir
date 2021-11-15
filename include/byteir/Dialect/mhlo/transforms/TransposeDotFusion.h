//===- TransposeDotFusion.h -----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_TRANSPOSEDOTFUSION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_TRANSPOSEDOTFUSION_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

void populateTransposeDotToDotGeneralPattern(RewritePatternSet &patterns);

std::unique_ptr<FunctionPass> createTransposeDotFusionPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_TRANSPOSEDOTFUSION_H