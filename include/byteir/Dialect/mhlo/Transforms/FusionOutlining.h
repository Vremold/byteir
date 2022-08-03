//===- FusionOutlining.h --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_FUSIONOUTLINING_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_FUSIONOUTLINING_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;

std::unique_ptr<OperationPass<ModuleOp>> createFusionOutliningPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_FUSIONOUTLINING_H