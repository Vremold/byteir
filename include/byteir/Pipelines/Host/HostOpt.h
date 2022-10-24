//===- HostOpt.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_HOST_HOSTOPT_H
#define BYTEIR_PIPELINES_HOST_HOSTOPT_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
struct HostOptPipelineOptions
    : public PassPipelineOptions<HostOptPipelineOptions> {
  Option<std::string> fileName{
      *this, "file-name",
      llvm::cl::desc(
          "To specify where the generated llvm kernel will be writed to"),
      llvm::cl::init("host_kernels.ll")};
};

void createHostOptPipeline(OpPassManager &pm,
                           const HostOptPipelineOptions &options);

inline void registerHostOptPipeline() {
  PassPipelineRegistration<HostOptPipelineOptions>(
      "host-opt", "Host Opt Pipeline", createHostOptPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_HOST_HOSTOPT_H
