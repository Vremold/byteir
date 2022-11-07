//===- LinalgTensorOpt.h --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_LINALGTENSOROPT_H
#define BYTEIR_PIPELINES_LINALGTENSOROPT_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <memory>
#include <string>

namespace mlir {
struct LinalgTensorOptPipelineOptions
    : public PassPipelineOptions<LinalgTensorOptPipelineOptions> {
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("An optional attribute to speicify target."),
      llvm::cl::init("")};
};

void createLinalgTensorOptPipeline(
    OpPassManager &pm, const LinalgTensorOptPipelineOptions &options);

inline void registerLinalgTensorOptPipeline() {
  PassPipelineRegistration<LinalgTensorOptPipelineOptions>(
      "linalg-tensor-opt", "Linalg Opt Pipeline in Tensor",
      createLinalgTensorOptPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_LINALGTENSOROPT_H