//===- Util.h -------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_UTILUTIL_H
#define BYTEIR_DIALECT_MHLO_UTILUTIL_H

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <stdint.h>

namespace mlir {
class Attribute;
class Block;
class NamedAttrList;
class Operation;
class OpBuilder;
class Value;

// Return true, if op is a splat constant
bool IsSplatMhloConstant(Operation *op);

// Return true if op is either a splat constant, or another constant-like op like iota
bool IsMhloConstantLike(Operation *op);

bool IsSplatMhloConstantValue(Operation *op, int64_t splat_val);

bool IsSplatMhloConstantValue(Operation *op, double splat_val);

bool IsSplatMhloConstantValue(Value val, int64_t splat_val);

bool IsSplatMhloConstantValue(Value val, double splat_val);

bool IsBlockSingleAdd(Block *block);

template <typename T>
void HandleConvAttribute(NamedAttrList &attrs, T conv_op, OpBuilder &rewriter);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_UTILUTIL_H