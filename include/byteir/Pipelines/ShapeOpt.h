//===- ShapeOpt.h ------------------------------------------------- C++ ---===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_SHAPEOPT_H
#define BYTEIR_PIPELINES_SHAPEOPT_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

void createShapeOptPipeline(OpPassManager &pm);

inline void registerShapeOptPipeline() {
  PassPipelineRegistration<>("shape-opt", "Shape Opt Pipeline",
                             createShapeOptPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_SHAPEOPT_H
