//===- MhloPreprocessing.h ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_TRANSPOSEFOLDER_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_TRANSPOSEFOLDER_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

// Patterns to fold mhlo::TransposeOp
void populateFoldTransposePatterns(RewritePatternSet &patterns);

std::unique_ptr<FunctionPass> createTransposeFolderPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_TRANSPOSEFOLDER_H