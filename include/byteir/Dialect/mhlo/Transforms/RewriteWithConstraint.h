//===- RewriteWithConstraint.h --------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_REWRITEWITHCONSTRAINT_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_REWRITEWITHCONSTRAINT_H

#include "mlir/Pass/Pass.h"

namespace mlir {

void populateRewriteWithConstraintConstraintPattern(
    RewritePatternSet &patterns);
std::unique_ptr<OperationPass<FuncOp>> createRewriteWithConstraintPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_REWRITEWITHCONSTRAINT_H