//===- CollectFunc.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_COLLECTFUNC_H
#define BYTEIR_TRANSFORMS_COLLECTFUNC_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

std::unique_ptr<OperationPass<ModuleOp>>
createCollectFuncPass(llvm::StringRef anchorTag = "");

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_COLLECTFUNC_H