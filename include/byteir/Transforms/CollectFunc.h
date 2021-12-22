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
#include <string>

namespace mlir {

std::unique_ptr<OperationPass<ModuleOp>>
createCollectFuncPass(const std::string& anchorTag = "");

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_COLLECTFUNC_H