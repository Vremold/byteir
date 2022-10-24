//===- AffineOpt.h ------------------------------------------------ C++ ---===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_AFFINEOPT_H
#define BYTEIR_PIPELINES_AFFINEOPT_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

void createAffineOptPipeline(OpPassManager &pm);

inline void registerAffineOptPipeline() {
  PassPipelineRegistration<>("affine-opt", "Affine Opt Pipeline",
                             createAffineOptPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_AFFINEOPT_H
