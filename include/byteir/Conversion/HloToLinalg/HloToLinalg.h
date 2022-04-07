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

void populateUnrealizedCastToLinalgConversionPattern(
    MLIRContext *context, OwningRewritePatternList *patterns);

std::unique_ptr<OperationPass<FuncOp>>
createHloFusionToLinalgPass(llvm::StringRef anchorTag = "");

std::unique_ptr<OperationPass<FuncOp>> createUnrealizedCastToLinalgPass();

} // namespace mlir

#endif // BYTEIR_CONVERSION_HLOTOLINALG_H