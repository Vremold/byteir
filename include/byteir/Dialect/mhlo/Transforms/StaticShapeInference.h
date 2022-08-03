//===- StaticShapeInference.h ---------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_STATICSHAPEINFERENCE_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_STATICSHAPEINFERENCE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

std::unique_ptr<OperationPass<func::FuncOp>> createStaticShapeInferencePass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_STATICSHAPEINFERENCE_H