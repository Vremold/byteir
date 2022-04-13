//===- ArithTMhloPattern.td -----------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef ARITH_TO_MHLO_PATTERN
#define ARITH_TO_MHLO_PATTERN

include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"

def ArithToMhloConstOpPattern : Pat<
  (Arith_ConstantOp $value),
  (HLO_ConstOp $value)>;


#endif // ARITH_TO_MHLO_PATTERN