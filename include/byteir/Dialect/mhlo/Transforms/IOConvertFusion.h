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

void populateIOConvertBatchNormPattern(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<FuncOp>> createIOConvertFusionPass();

std::unique_ptr<OperationPass<FuncOp>>
createIOConvertFusionPass(std::string opName, std::string byreComputeName);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_BATCHNORMTRAININGFUSION_H