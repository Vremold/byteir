//===- HloFusionToLinalg.h ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_HLOTOLINALG_H
#define BYTEIR_CONVERSION_HLOTOLINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

void populateUnrealizedCastToLinalgConversionPattern(
    MLIRContext *context, RewritePatternSet *patterns);

std::unique_ptr<OperationPass<func::FuncOp>>
createHloFusionToLinalgPass(llvm::StringRef anchorTag = "");

std::unique_ptr<OperationPass<func::FuncOp>> createUnrealizedCastToLinalgPass();

} // namespace mlir

#endif // BYTEIR_CONVERSION_HLOTOLINALG_H