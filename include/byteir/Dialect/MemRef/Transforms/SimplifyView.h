//===- SimplifyView.h -----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MEMREF_TRANSFORMS_SIMPLIFYVIEW_H
#define BYTEIR_DIALECT_MEMREF_TRANSFORMS_SIMPLIFYVIEW_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

void populateSimplifyViewPattern(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<func::FuncOp>> createSimplifyViewPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MEMREF_TRANSFORMS_SIMPLIFYVIEW_H