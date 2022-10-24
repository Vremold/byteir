//===- AllOpt.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_ALLOPT_H
#define BYTEIR_PIPELINES_ALLOPT_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
struct ByteIRAllOptPipelineOptions
    : public PassPipelineOptions<ByteIRAllOptPipelineOptions> {
  Option<std::string> entryFunc{
      *this, "entry-func",
      llvm::cl::desc("An optional string to speicify entry function."),
      llvm::cl::init("main")};
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("An optional attribute to speicify target."),
      llvm::cl::init("")};
};

void createByteIRAllOptPipeline(OpPassManager &pm,
                                const ByteIRAllOptPipelineOptions &options);

inline void registerByteIRAllOptPipeline() {
  PassPipelineRegistration<ByteIRAllOptPipelineOptions>(
      "byteir-all-opt", "Byteir all Opt Pipeline", createByteIRAllOptPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_ALLOPT_H