//===- ConvBiasActFusion.h ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVBIASACTFUSION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVBIASACTFUSION_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

void populateFuseConvBiasActPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<FuncOp>> createConvBiasActFusionPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVBIASACTFUSION_H