//===- BatchNormTrainingFusion.h ------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_BATCHNORMTRAININGFUSION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_BATCHNORMTRAININGFUSION_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>
#include <vector>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

void populateIOConvertBatchNormPattern(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<func::FuncOp>> createIOConvertFusionPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_BATCHNORMTRAININGFUSION_H