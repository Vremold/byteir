//===- HloOpt.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_HLOOPT_H
#define BYTEIR_PIPELINES_HLOOPT_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include <string>

namespace mlir {
struct HloOptPipelineOptions
    : public PassPipelineOptions<HloOptPipelineOptions> {
  Option<std::string> entryFunc{
      *this, "entry-func",
      llvm::cl::desc("An optional string to speicify entry function."),
      llvm::cl::init("main")};
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("An optional attribute to speicify target."),
      llvm::cl::init("")};
  Option<bool> outlineSingleElemwiseOp{
      *this, "outline-single-elemwise-op",
      llvm::cl::desc("whether to outline the single element-wise operation as "
                     "an independent function"),
      llvm::cl::init(false)};
};

void createHloOptPipeline(OpPassManager &pm,
                          const HloOptPipelineOptions &options);

inline void registerHloOptPipeline() {
  PassPipelineRegistration<HloOptPipelineOptions>("hlo-opt", "Hlo Opt Pipeline",
                                                  createHloOptPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_HLOOPT_H
