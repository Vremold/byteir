//===- PassDetail.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_TO_PASSDETAIL_H
#define BYTEIR_CONVERSION_TO_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

// forward dialects for conversions
namespace mlir {
class AffineDialect;

namespace ace {
class AceDialect;
} // namespace ace

namespace byre {
class ByreDialect;
} // namespace byre

namespace gpu {
class GPUDialect;
class GPUModuleOp;
} // namespace byre

namespace lmhlo {
class LmhloDialect;
} // namespace lmhlo

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace memref {
class MemRefDialect;
} // namespace memref

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