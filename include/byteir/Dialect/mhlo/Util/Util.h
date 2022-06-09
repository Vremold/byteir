//===- Util.h -------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_MHLO_UTIL_UTIL_H
#define BYTEIR_MHLO_UTIL_UTIL_H

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

bool isMhlo(Operation *op);

bool isSplatMhloConstant(Operation *op);

// Return true if op is either a splat constant, or another constant-like op
// like iota
bool isSplatMhloConstantLike(Operation *op);

bool isMhloConstantLike(Operation *op);

bool isSplatMhloConstantValue(Operation *op, int64_t splat_val);

bool isSplatMhloConstantValue(Operation *op, double splat_val);

bool isSplatMhloConstantValue(Value val);

bool isSplatMhloConstantValue(Value val, int64_t splat_val);

bool isSplatMhloConstantValue(Value val, double splat_val);

// Return ture if block contains single op: AddOp, MaxOp, MinOp
template <typename Op> bool isBlockSingleOp(Block *block);

// Return layout like "NCHW"/"NHWC"/"NDHWC"/"NCDHW" if success,
// return "UNKNOWN" if failed.
std::string getPoolLayout(mhlo::ReduceWindowOp op);

// Return layout like "NCHW"/"NHWC"/"NDHWC"/"NCDHW" if success,
// return "UNKNOWN" if failed.
std::string getPoolGradLayout(mhlo::SelectAndScatterOp op);

// Return {input_layout, kernel_layout, output_layout} like PoolLayout,
// layout could be "NCHW"/"NHWC"/"HWCN"/"NDHWC"/"NCDHW"/"DHWCN",
// return "UNKNOWN" if failed.
std::tuple<std::string, std::string, std::string>
getConvLayout(mhlo::ConvDimensionNumbersAttr dimension_numbers);

template <typename T>
void handleConvAttribute(NamedAttrList &attrs, T conv_op, OpBuilder &rewriter);

} // namespace mlir

#endif // BYTEIR_MHLO_UTIL_UTIL_H