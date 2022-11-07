//===- ToLLVM.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_HOST_TOLLVM_H
#define BYTEIR_PIPELINES_HOST_TOLLVM_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

void createToLLVMPipeline(OpPassManager &pm);

inline void registerToLLVMPipeline() {
  PassPipelineRegistration<>("to-llvm", "To LLVM dialect Pipeline",
                             createToLLVMPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_HOST_TOLLVM_H
