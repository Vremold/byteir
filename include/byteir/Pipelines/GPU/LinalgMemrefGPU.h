//===- LinalgMemrefGPU.h --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_GPU_LINALGMEMREFGPU_H
#define BYTEIR_PIPELINES_GPU_LINALGMEMREFGPU_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

struct LinalgMemrefGPUPipelineOptions
    : public PassPipelineOptions<LinalgMemrefGPUPipelineOptions> {
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("An optional attribute to speicify target."),
      llvm::cl::init("")};
};

void createLinalgMemrefGPUPipeline(
    OpPassManager &pm, const LinalgMemrefGPUPipelineOptions &options);

inline void registerLinalgMemrefGPUPipeline() {
  PassPipelineRegistration<LinalgMemrefGPUPipelineOptions>(
      "linalg-memref-gpu", "Linalg Opt Pipeline in Memref for GPU",
      createLinalgMemrefGPUPipeline);
}

struct MatmulEpilogueGPUPipelineOptions
    : public PassPipelineOptions<MatmulEpilogueGPUPipelineOptions> {
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("An optional attribute to speicify target."),
      llvm::cl::init("")};
};

void createMatmulEpilogueGPUPipeline(
    OpPassManager &pm, const MatmulEpilogueGPUPipelineOptions &options);

inline void registerMatmulEpilogueGPUPipeline() {
  PassPipelineRegistration<MatmulEpilogueGPUPipelineOptions>(
      "matmul-epilogue-gpu", "Mamtmul Epilogue for gpu",
      createMatmulEpilogueGPUPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_GPU_LINALGMEMREFGPU_H
