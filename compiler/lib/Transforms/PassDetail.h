//===- PassDetail.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_PASSDETAIL_H
#define BYTEIR_TRANSFORMS_PASSDETAIL_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

// forward dialects for conversions
namespace mlir {
class ModuleOp;
class AffineDialect;

namespace cf {
class ControlFlowDialect;
} // namespace cf

namespace func {
class FuncDialect;
class FuncOp;
} // namespace func

namespace memref {
class MemRefDialect;
} // namespace memref

namespace mhlo {
class MhloDialect;
} // namespace mhlo

namespace scf {
class SCFDialect;
} // namespace scf

#define GEN_PASS_CLASSES
#include "byteir/Transforms/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_PASSDETAIL_H