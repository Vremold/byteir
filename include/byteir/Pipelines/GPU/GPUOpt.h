//===- GPUOpt.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_GPUOPT_H
#define BYTEIR_PIPELINES_GPUOPT_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include <string>

namespace mlir {
struct GPUOptPipelineOptions
    : public PassPipelineOptions<GPUOptPipelineOptions> {
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("An optional attribute to speicify target."),
      llvm::cl::init("")};
};

void createGPUOptPipeline(OpPassManager &pm,
                          const GPUOptPipelineOptions &options);

inline void registerGPUOptPipeline() {
  PassPipelineRegistration<GPUOptPipelineOptions>("gpu-opt", "GPU Opt Pipeline",
                                                  createGPUOptPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_GPUOPT_H
