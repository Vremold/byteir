//===- PassDetail.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_HOST_PASSDETAIL_H
#define BYTEIR_PIPELINES_HOST_PASSDETAIL_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace arith {
class ArithmeticDialect;
} // namespace arith

namespace bufferization {
class BufferizationDialect;
} // namespace bufferization

namespace cf {
class ControlFlowDialect;
} // namespace cf

namespace func {
class FuncDialect;
} // namespace func

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

namespace memref {
class MemRefDialect;
} // namespace memref

namespace scf {
class SCFDialect;
} // namespace scf

namespace tensor {
class TensorDialect;
} // namespace tensor

#define GEN_PASS_CLASSES
#include "byteir/Pipelines/Host/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_PIPELINES_HOST_PASSDETAIL_H
