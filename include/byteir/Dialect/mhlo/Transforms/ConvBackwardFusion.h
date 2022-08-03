//===- ConvBackwardFusion.h -----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVBACKWARDFUSION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVBACKWARDFUSION_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class RewritePatternSet;
namespace func {
class FuncOp;
} // namespace func

void populateFuseConvBackwardPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<func::FuncOp>> createConvBackwardFusionPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVBACKWARDFUSION_H