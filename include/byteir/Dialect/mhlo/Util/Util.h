//===- Util.h -------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_UTILUTIL_H
#define BYTEIR_DIALECT_MHLO_UTILUTIL_H

#include <stdint.h>

namespace mlir {
class Block;
class Operation;
class Value;

bool IsSplatMhloConstant(Operation* op);

bool IsSplatMhloConstantValue(Operation* op, int64_t splat_val);

bool IsSplatMhloConstantValue(Operation* op, double splat_val);

bool IsSplatMhloConstantValue(Value val, int64_t splat_val);

bool IsSplatMhloConstantValue(Value val, double splat_val);

bool IsBlockSingleAdd(Block* block);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_UTILUTIL_H