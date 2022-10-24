//===- ByreOpt.h -----------------------------------------------*--- C++
//-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_BYREOPT_H
#define BYTEIR_PIPELINES_BYREOPT_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include <string>

namespace mlir {
struct ByreOptPipelineOptions
    : public PassPipelineOptions<ByreOptPipelineOptions> {
  Option<std::string> entryFunc{
      *this, "entry-func",
      llvm::cl::desc("An optional string to speicify entry function."),
      llvm::cl::init("main")};
  Option<bool> appendArgTypes{
      *this, "append-arg-types",
      llvm::cl::desc("whether to append arg types to Byre"),
      llvm::cl::init(false)};
  Option<bool> disableMemoryPlanning{
      *this, "disable-memory-planning",
      llvm::cl::desc("whether to disable memory planning"),
      llvm::cl::init(false)};
};

void createByreOptPipeline(OpPassManager &pm,
                           const ByreOptPipelineOptions &options);

inline void registerByreOptPipeline() {
  PassPipelineRegistration<ByreOptPipelineOptions>(
      "byre-opt", "Byre Opt Pipeline", createByreOptPipeline);
}
} // namespace mlir

#endif // BYTEIR_PIPELINES_BYREOPT_H