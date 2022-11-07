//===- ArithTMhloPattern.td -----------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef ARITH_TO_MHLO_PATTERN
#define ARITH_TO_MHLO_PATTERN

include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"

def ArithToMhloConstOpPattern : Pat<
  (Arith_ConstantOp:$output $value),
  (HLO_ConstantOp $value),
  [(HLO_Tensor $output)]>;


#endif // ARITH_TO_MHLO_PATTERN