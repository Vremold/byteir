//===- NVVMCodegen.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_NVVMCODEGEN_H
#define BYTEIR_PIPELINES_NVVMCODEGEN_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

void createNVVMCodegenPipeline(OpPassManager &pm);

inline void registerNVVMCodegenPipeline() {
  PassPipelineRegistration<>("nvvm-codegen", "NVVM Codegen Pipeline",
                             createNVVMCodegenPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_NVVMCODEGEN_H
