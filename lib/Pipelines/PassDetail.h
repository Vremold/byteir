//===- PassDetail.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_PASSDETAIL_H
#define BYTEIR_PIPELINES_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class AffineDialect;

namespace arith {
class ArithmeticDialect;
} // namespace arith

namespace bufferization {
class BufferizationDialect;
} // namespace bufferization

namespace byre {
class ByreDialect;
} // namespace byre

namespace cf {
class ControlFlowDialect;
} // namespace cf

namespace func {
class FuncDialect;
} // namespace func

namespace gpu {
class GPUDialect;
} // namespace gpu

namespace lace {
class LaceDialect;
} // namespace lace

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace lmhlo {
class LmhloDialect;
} // namespace lmhlo

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
#include "byteir/Pipelines/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_PIPELINES_PASSDETAIL_H
