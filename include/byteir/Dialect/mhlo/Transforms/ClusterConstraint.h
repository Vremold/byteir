//===- ClusterConstraint.h ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_CLUSTER_CONSTRAINT_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_CLUSTER_CONSTRAINT_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

void populateClusterConstraintPattern(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<func::FuncOp>> createClusterConstraintPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_CLUSTER_CONSTRAINT_H