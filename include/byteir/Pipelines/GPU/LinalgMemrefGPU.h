//===- LinalgMemrefGPU.h --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_GPU_LINALGMEMREFGPU_H
#define BYTEIR_PIPELINES_GPU_LINALGMEMREFGPU_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <memory>
#include <string>

namespace mlir {
class ModuleOp;

std::unique_ptr<OperationPass<ModuleOp>>
createMatmulEpilogueGPUPipelinePass(const std::string &target = "");

std::unique_ptr<OperationPass<ModuleOp>>
createLinalgMemrefGPUPipelinePass(const std::string &target = "");

} // namespace mlir

#endif // BYTEIR_PIPELINES_GPU_LINALGMEMREFGPU_H
