//===- BoundedShapeInference.h --------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_BOUNDEDSHAPEINFERENCE_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_BOUNDEDSHAPEINFERENCE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

constexpr StringRef getBoundedShapeAttrName() { return "byteir.bounded_shape"; }

std::unique_ptr<OperationPass<FuncOp>> createBoundedShapeInferencePass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_BOUNDEDSHAPEINFERENCE_H