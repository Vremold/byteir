//===- FuncTag.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_FUNCTAG_H
#define BYTEIR_TRANSFORMS_FUNCTAG_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {

std::unique_ptr<OperationPass<ModuleOp>>
createFuncTagPass(const std::string& attachTag = "", const std::string& funcName = "");

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_FUNCTAG_H