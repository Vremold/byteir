//===- NVVMCodegen.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_NVVMCODEGEN_H
#define BYTEIR_PIPELINES_NVVMCODEGEN_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;

std::unique_ptr<OperationPass<ModuleOp>> createNVVMCodegenPipelinePass();

} // namespace mlir

#endif // BYTEIR_PIPELINES_NVVMCODEGEN_H
