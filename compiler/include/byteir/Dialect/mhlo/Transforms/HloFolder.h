//===- HloFolder.h -------------------------------------------*--- C++ -*-===//
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
namespace func {
class FuncOp;
} // namespace func

// Patterns to fold hlo ops
void populateHloFoldPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<func::FuncOp>> createHloFolderPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_HLOFOLDER_H