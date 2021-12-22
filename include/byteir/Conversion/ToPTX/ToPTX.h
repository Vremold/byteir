//===- ToPTX.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_TOPTX_H
#define BYTEIR_CONVERSION_TOPTX_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {

std::unique_ptr<FunctionPass> createGenPTXConfigPass();

// TODO move to general GPU
std::unique_ptr<OperationPass<ModuleOp>> createCollectGPUKernelPass(const std::string& name = "unified");

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOPTX_H