//===- Util.cpp -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

using namespace mlir;
using namespace llvm;

bool mlir::IsSplatMhloConstant(Operation* op) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstOp>(op)) {
    return constOp.value().isSplat();
  }
  return false;
}

bool mlir::IsSplatMhloConstantValue(Operation* op, int64_t splat_val) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstOp>(op)) {
    // only handle DenseFPElementsAttr for now
    // TODO extend it 
    if (auto denseIntE = constOp.value().dyn_cast<DenseIntElementsAttr>()) {
      return isSplatValue(denseIntE, splat_val);
    }
  }
  return false;
}

bool mlir::IsSplatMhloConstantValue(Operation* op, double splat_val) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstOp>(op)) {
    // only handle DenseFPElementsAttr for now
    // TODO extend it 
    if (auto denseFPE = constOp.value().dyn_cast<DenseFPElementsAttr>()) {
      return isSplatValue(denseFPE, splat_val);
    }
  }
  return false;
}


bool mlir::IsSplatMhloConstantValue(Value val, int64_t splat_val) {
  return IsSplatMhloConstantValue(val.getDefiningOp(), splat_val);
}

bool mlir::IsSplatMhloConstantValue(Value val, double splat_val) {
  return IsSplatMhloConstantValue(val.getDefiningOp(), splat_val);
}

// TODO: make this a template later for max/min
bool mlir::IsBlockSingleAdd(Block* block) {
  if (block == nullptr) return false;

  auto ret_op = block->getTerminator();
  if (!isa<mlir::mhlo::ReturnOp>(ret_op)) return false;

  auto mhlo_ret = cast<mlir::mhlo::ReturnOp>(ret_op);
  if (mhlo_ret.getNumOperands() != 1) return false;

  auto compute_op = mhlo_ret.getOperand(0).getDefiningOp();
  if (auto add_op = dyn_cast_or_null<mhlo::AddOp>(compute_op)) {
    return (compute_op->getOperand(0) == block->getArgument(0) &&
            compute_op->getOperand(1) == block->getArgument(1)) ||
           (compute_op->getOperand(0) == block->getArgument(1) &&
            compute_op->getOperand(1) == block->getArgument(0));
  }

  return false;
}
