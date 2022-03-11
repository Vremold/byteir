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

void populateClusterConstraintPattern(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<FuncOp>> createClusterConstraintPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_CLUSTER_CONSTRAINT_H