//===- ByreHost.h ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_BYREHOST_H
#define BYTEIR_PIPELINES_BYREHOST_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include <string>

namespace mlir {
struct ByreHostPipelineOptions
    : public PassPipelineOptions<ByreHostPipelineOptions> {
  Option<std::string> entryFunc{
      *this, "entry-func",
      llvm::cl::desc("An optional string to speicify entry function."),
      llvm::cl::init("main")};
  Option<std::string> deviceFile{
      *this, "device-file-name",
      llvm::cl::desc("An optional string to speicify device file name."),
      llvm::cl::init("kernel")};
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("An optional attribute to target device."),
      llvm::cl::init("")};
};

void createByreHostPipeline(OpPassManager &pm,
                            const ByreHostPipelineOptions &options);

inline void registerByreHostPipeline() {
  PassPipelineRegistration<ByreHostPipelineOptions>(
      "byre-host", "Byre Host Pipeline", createByreHostPipeline);
}
} // namespace mlir

#endif // BYTEIR_PIPELINES_BYREHOST_H