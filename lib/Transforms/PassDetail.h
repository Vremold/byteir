//===- PassDetail.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_PASSDETAIL_H
#define BYTEIR_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

// forward dialects for conversions
namespace mlir {

class StandardOpsDialect;

namespace memref {
class MemRefDialect;
} // namespace memref

#define GEN_PASS_CLASSES
#include "byteir/Transforms/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_PASSDETAIL_H
