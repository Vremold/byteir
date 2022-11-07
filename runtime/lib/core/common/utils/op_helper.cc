//===- op_helper.cc -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/core/common/utils/op_helper.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace mlir;

namespace brt {

bool IsLocalAlias(Operation *op) {
  if (!IsAliasOp(op))
    return false;
  return op->getOperand(0).isa<mlir::OpResult>();
}

bool IsArgAlias(Operation *op) {
  if (!IsAliasOp(op))
    return false;
  return op->getOperand(0).isa<mlir::BlockArgument>();
}

bool IsAliasOp(Operation *op) { return llvm::isa<mlir::byre::AliasOp>(op); }

// FIXME: How to handle i1 alias offset
size_t GetAliasOffsetInByte(Operation *op) {
  auto offset = llvm::cast<mlir::byre::AliasOp>(op).getOffset();
  if (auto memref = op->getOperand(0).getType().dyn_cast<mlir::MemRefType>()) {
    unsigned int element_byte = GetElementTypeByte(memref);
    return static_cast<size_t>(offset) * static_cast<size_t>(element_byte);
  }

  return 0;
}

bool IsAllocOp(Operation *op) {
  if (auto iface = llvm::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(op)) {
    if (iface.hasEffect<MemoryEffects::Allocate>()) {
      return true;
    }
  }
  return false;
}

bool IsDynamicAllocOp(Operation *op, std::vector<mlir::Value> &dynamicSizes) {
  if (IsAllocOp(op)) {
    if (auto memref = op->getResult(0).getType().dyn_cast<MemRefType>()) {
      if (!memref.hasStaticShape()) {
        // TODO: this depends on the dynamic sizes are the first N operands
        dynamicSizes.insert(dynamicSizes.end(), op->operand_begin(),
                            op->operand_begin() + memref.getNumDynamicDims());
        return true;
      }
    }
  }
  return false;
}

bool IsShapeComputeOp(Operation *op) {
  return llvm::isa<byre::ComputeShapeOp>(op);
}

} // namespace brt
