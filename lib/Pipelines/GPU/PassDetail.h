//===- PassDetail.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_GPU_PASSDETAIL_H
#define BYTEIR_PIPELINES_GPU_PASSDETAIL_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class AffineDialect;

namespace arith {
class ArithmeticDialect;
} // namespace arith

namespace cf {
class ControlFlowDialect;
} // namespace cf

namespace func {
class FuncDialect;
} // namespace func

namespace gpu {
class GPUDialect;
} // namespace gpu

namespace memref {
class MemRefDialect;
} // namespace memref

namespace NVVM {
class NVVMDialect;
} // namespace NVVM

namespace scf {
class SCFDialect;
} // namespace scf

#define GEN_PASS_CLASSES
#include "byteir/Pipelines/GPU/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_PIPELINES_GPU_PASSDETAIL_H
