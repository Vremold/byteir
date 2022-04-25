//===- PassDetail.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_SCF_TRANSFORMS_PASSDETAIL_H
#define BYTEIR_DIALECT_SCF_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

// forward dialects for conversions
namespace mlir {
namespace scf {
class SCFDialect;
} // namespace scf

#define GEN_PASS_CLASSES
#include "byteir/Dialect/SCF/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_DIALECT_SCF_TRANSFORMS_PASSDETAIL_H
