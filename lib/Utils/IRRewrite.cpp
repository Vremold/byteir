//===- IRRewrite.cpp ----------------------------------- -----------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/Utils.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <tuple>

using namespace llvm;
using namespace mlir;

void mlir::ReplicateDefiningOp(Block* block, std::function<bool(Operation*)> checkFunc) {
  if (block->empty()) return;
  auto ctx = block->front().getContext();
  OpBuilder builder(ctx);

  SmallVector<std::tuple<Operation*, unsigned int, unsigned int>> replaceOps;

  for (auto it = block->begin(); it != block->end(); ++it) {
    auto& op = *it;
   
    for (unsigned int i = 0; i < op.getNumOperands(); ++i) {
      auto val = op.getOperand(i);
      auto opDef = val.getDefiningOp();
      if (opDef != nullptr && checkFunc(opDef)) {
        auto maybeIdx = FindResultIndex(opDef, val);
        replaceOps.emplace_back(&op, i, maybeIdx.getValue());
      }
    }
  }

  for (auto& t : replaceOps) {
    auto op = std::get<0>(t);
    auto opId = std::get<1>(t);
    auto resId = std::get<2>(t);
    auto opDef = op->getOperand(opId).getDefiningOp();

    builder.setInsertionPoint(opDef);
    auto cloned = builder.clone(*opDef);
    op->setOperand(opId, cloned->getResult(resId));
  }
}
