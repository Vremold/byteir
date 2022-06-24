//===- PassDetail.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_SHAPE_TRANSFORMS_PASSDETAIL_H
#define BYTEIR_DIALECT_SHAPE_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

// forward dialects for conversions
namespace mlir {

namespace tensor {
class TensorDialect;
} // namespace tensor

namespace shape {
class ShapeDialect;
} // namespace shape

namespace shape_ext {
class ShapeExtDialect;
} // namespace shape_ext

#define GEN_PASS_CLASSES
#include "byteir/Dialect/Shape/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_DIALECT_SHAPE_TRANSFORMS_PASSDETAIL_H
