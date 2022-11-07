//===- SCFOpt.h --------------------------------------------------- C++ ---===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_SCFOPT_H
#define BYTEIR_PIPELINES_SCFOPT_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

void createSCFOptPipeline(OpPassManager &pm);

inline void registerSCFOptPipeline() {
  PassPipelineRegistration<>("scf-opt", "SCF Opt Pipeline",
                             createSCFOptPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_SCFOPT_H
