//===- AffineLoopFusionEx.h ----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_AFFINE_TRANSFORMS_AFFINELOOPFUSIONEX_H
#define BYTEIR_DIALECT_AFFINE_TRANSFORMS_AFFINELOOPFUSIONEX_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

std::unique_ptr<OperationPass<FuncOp>>
createAffineLoopFusionExPass(const std::string &attachTag = "");

} // namespace mlir

#endif // BYTEIR_DIALECT_AFFINE_TRANSFORMS_AFFINELOOPFUSIONEX_H