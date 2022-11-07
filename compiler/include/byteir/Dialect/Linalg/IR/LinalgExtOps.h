//===- ShapeExtOps.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_LINALGEXTOPS_H
#define BYTEIR_DIALECT_LINALG_LINALGEXTOPS_H

#include "byteir/Dialect/Linalg/IR/LinalgExtInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

// some util func
namespace mlir {
namespace linalg_ext {
//
/// Returns a `memref.dim` or `tensor.dim` operation to get the shape of `v` at
/// `dim`.
Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim);

/// Returns a `memref.dim` or `tensor.dim` operation to get the shape of `v` at
/// `dim`. If the shape is constant, returns the shape as an `IntegerAttr`.
OpFoldResult getDim(OpBuilder &builder, Location loc, Value v, int64_t dim);

} // namespace linalg_ext
} // namespace mlir

#include "byteir/Dialect/Linalg/IR/LinalgExtOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h.inc"

#endif // BYTEIR_DIALECT_LINALG_LINALGEXTOPS_H
