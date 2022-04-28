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
#include <string>
#include <tuple>

namespace mlir {
class Attribute;
class Block;
class NamedAttrList;
class Operation;
class OpBuilder;
class Value;

bool IsSplatMhloConstant(Operation *op);

// Return true if op is either a splat constant, or another constant-like op
// like iota
bool IsSplatMhloConstantLike(Operation *op);

bool IsMhloConstantLike(Operation *op);

bool IsSplatMhloConstantValue(Operation *op, int64_t splat_val);

bool IsSplatMhloConstantValue(Operation *op, double splat_val);

bool IsSplatMhloConstantValue(Value val);

bool IsSplatMhloConstantValue(Value val, int64_t splat_val);

bool IsSplatMhloConstantValue(Value val, double splat_val);

bool IsBlockSingleAdd(Block *block);

// Return layout like "NCHW"/"NHWC"/"NDHWC"/"NCDHW" if success,
// Return "UNKNOWN" if failed.
std::string GetPoolLayout(mhlo::ReduceWindowOp op);

// Return {input_layout, kernel_layout, output_layout} like PoolLayout
std::tuple<std::string, std::string, std::string>
GetConvLayout(mhlo::ConvDimensionNumbersAttr dimension_numbers);

template <typename T>
void HandleConvAttribute(NamedAttrList &attrs, T conv_op, OpBuilder &rewriter);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_UTILUTIL_H