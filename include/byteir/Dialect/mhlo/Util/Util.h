//===- Util.h -------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_UTIL_UTIL_H
#define BYTEIR_DIALECT_MHLO_UTIL_UTIL_H

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

namespace byteir {

enum class NamedLayout : uint32_t {
  UNKNOWN = 0,
  NHWC = 1,
  NDHWC = 2,
  NCHW = 3,
  NCDHW = 4,
  HWCN = 5,
  DHWCN = 6,
  NCL = 7,
};

inline std::string stringifyEnum(NamedLayout layout) {
  switch (layout) {
  case NamedLayout::UNKNOWN:
    return "UNKNOWN";
  case NamedLayout::NHWC:
    return "NHWC";
  case NamedLayout::NDHWC:
    return "NDHWC";
  case NamedLayout::NCHW:
    return "NCHW";
  case NamedLayout::NCDHW:
    return "NCDHW";
  case NamedLayout::HWCN:
    return "HWCN";
  case NamedLayout::DHWCN:
    return "DHWCN";
  case NamedLayout::NCL:
    return "NCL";
  default:
    return "UNKNOWN";
  }
}

} // namespace byteir

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

// Return layout if success, return UNKNOWN if failed.
byteir::NamedLayout getPoolLayout(mhlo::ReduceWindowOp op);

// Return layout if success, return "UNKNOWN" if failed.
byteir::NamedLayout getPoolGradLayout(mhlo::SelectAndScatterOp op);

// Return {input_layout, kernel_layout, output_layout} like PoolLayout,
// return UNKNOWN if failed.
std::tuple<byteir::NamedLayout, byteir::NamedLayout, byteir::NamedLayout>
getConvLayout(mhlo::ConvDimensionNumbersAttr dimension_numbers);

template <typename T>
void handleConvAttribute(NamedAttrList &attrs, T conv_op, OpBuilder &rewriter);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_UTIL_UTIL_H
