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
class AffineDialect;

namespace cf {
class ControlFlowDialect;
} // namespace cf

namespace func {
class FuncDialect;
} // namespace func

namespace memref {
class MemRefDialect;
} // namespace memref

namespace scf {
class SCFDialect;
} // namespace scf

#define GEN_PASS_CLASSES
#include "byteir/Transforms/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_PASSDETAIL_H
