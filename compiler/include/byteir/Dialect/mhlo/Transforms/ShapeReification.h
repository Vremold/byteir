//===- ShapeReification.h -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_SHAPEREIFICATION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_SHAPEREIFICATION_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

std::unique_ptr<OperationPass<func::FuncOp>> createByteIRShapeReificationPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_SHAPEREIFICATION_H
