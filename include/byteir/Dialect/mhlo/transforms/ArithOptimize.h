//===- ArithOptimize.h ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_ARITHOPTIMIZE_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_ARITHOPTIMIZE_H

#include "mlir/Pass/Pass.h"

namespace mlir {

void populateMhloArithOptPatterns(RewritePatternSet &patterns);
std::unique_ptr<FunctionPass> createMhloArithOptPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_ELEMENTFUSION_H