//===- GPUOpt.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_GPUOPT_H
#define BYTEIR_PIPELINES_GPUOPT_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {

std::unique_ptr<OperationPass<ModuleOp>>
createGPUOptPipelinePass(const std::string &target = "");

} // namespace mlir

#endif // BYTEIR_PIPELINES_GPUOPT_H
