//===- TransformInsertion.h -----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_TRANSFORM_INSERTION_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_TRANSFORM_INSERTION_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;

std::unique_ptr<OperationPass<ModuleOp>> createTransformInsertionPass(
    std::string deviceAnchorName = "__byteir_tiling_test__");

} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_TRANSFORM_INSERTION_H