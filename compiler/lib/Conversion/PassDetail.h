//===- PassDetail.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_TO_PASSDETAIL_H
#define BYTEIR_CONVERSION_TO_PASSDETAIL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

// forward dialects for conversions
namespace mlir {
class AffineDialect;

namespace ace {
class AceDialect;
} // namespace ace

namespace arith {
class ArithDialect;
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
class FuncOp;
} // namespace func

namespace gpu {
class GPUDialect;
class GPUModuleOp;
} // namespace gpu

namespace lace {
class LaceDialect;
} // namespace lace

namespace lmhlo {
class LmhloDialect;
} // namespace lmhlo

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace memref {
class MemRefDialect;
} // namespace memref

namespace mhlo {
class MhloDialect;
} // namespace mhlo

namespace NVVM {
class NVVMDialect;
} // namespace NVVM

namespace scf {
class SCFDialect;
} // namespace scf

namespace shape {
class ShapeDialect;
} // namespace shape

#define GEN_PASS_CLASSES
#include "byteir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_CONVERSION_TO_PASSDETAIL_H