//===- MhloPreprocessing.h ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_HLOFOLDER_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_HLOFOLDER_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

// Patterns to fold mhlo::TransposeOp
void populateHloFoldPatterns(RewritePatternSet &patterns);

std::unique_ptr<FunctionPass> createHloFolderPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_HLOFOLDER_H