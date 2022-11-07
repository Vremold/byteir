//===- LinalgExtInterfaces.h ----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_LINALGEXTINTERFACES_H
#define BYTEIR_DIALECT_LINALG_LINALGEXTINTERFACES_H

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace linalg_ext {
class LinalgExtOp;

namespace detail {
LogicalResult verifyLinalgExtOpInterface(Operation *op);
}

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h.inc" // IWYU pragma: export

/// Include the generated interface declarations.
#include "byteir/Dialect/Linalg/IR/LinalgExtOpInterfaces.h.inc" // IWYU pragma: export

} // namespace linalg_ext
} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_LINALGEXTINTERFACES_H
