//===- PassDetail.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_BYRE_TRANSFORMS_PASSDETAIL_H
#define BYTEIR_DIALECT_BYRE_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

// forward dialects for conversions
namespace mlir {

#define GEN_PASS_CLASSES
#include "byteir/Dialect/Byre/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_DIALECT_BYRE_TRANSFORMS_PASSDETAIL_H
