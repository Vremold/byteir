//===- ShapeOpt.h ------------------------------------------------- C++ ---===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_SHAPEOPT_H
#define BYTEIR_PIPELINES_SHAPEOPT_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

std::unique_ptr<OperationPass<ModuleOp>> createShapeOptPipelinePass();

} // namespace mlir

#endif // BYTEIR_PIPELINES_SHAPEOPT_H