//===- Fold.cpp -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Byre/Transforms/Fold.h"
#include "PassDetail.h"
#include "byteir/Analysis/Alias.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include <functional>

using namespace byteir;
using namespace llvm;
using namespace mlir;
using namespace mlir::byre;

namespace {

struct ByreAliasAnalysis : public AliasAnalysis {
  ByreAliasAnalysis(mlir::Block *b, llvm::ArrayRef<mlir::Value> initials,
                    std::function<bool(mlir::Operation &op)> checkAlias)
      : AliasAnalysis(b, initials, checkAlias) {
    offsets.resize(values.size(), 0);
  }

  int getOrCreateIndex(mlir::Value val) override {
    if (valueToIndex.count(val) == 0) {
      int count = values.size();
      valueToIndex[val] = count;
      values.push_back(val);
      leaderToIndex.insert(count);
      offsets.push_back(0);
    }
    return valueToIndex[val];
  }

  void runOnBlock() override {
    if (block->empty())
      return;

    for (auto &op : block->without_terminator()) {
      if (isAlias(op)) {
        int in_idx = getOrCreateIndex(op.getOperand(0));
        int in_leader = leaderToIndex.getLeaderValue(in_idx);
        int out_idx = getOrCreateIndex(op.getOperand(1));
        int out_leader = leaderToIndex.getLeaderValue(out_idx);

        if (in_leader <= out_idx) {
          leaderToIndex.unionSets(in_leader, out_leader);
        } else {
          leaderToIndex.unionSets(out_leader, in_leader);
        }

        int in_offset = offsets[in_idx];
        int op_offset = op.getAttrOfType<IntegerAttr>("offset").getInt();
        int out_offset = in_offset + op_offset;
        offsets[out_idx] = out_offset;
      }
    }
  }

  // track offset
  SmallVector<int> offsets;
};

static bool isAliasOp(Operation &op) {
  if (auto compute_op = dyn_cast<byre::ComputeOp>(op)) {
    return compute_op.getCallee() == "AliasOp";
  }
  return false;
};

static void foldAlias(func::FuncOp func) {
  auto ctx = func.getContext();
  // use all args as initials for alias
  SmallVector<Value> initial_copy;
  for (auto val : func.getArguments()) {
    initial_copy.push_back(val);
  }

  auto &func_block = func.getBody().front();
  ByreAliasAnalysis byre_alias(&func_block, initial_copy, isAliasOp);
  byre_alias.runOnBlock();

  SmallVector<ComputeOp> remove_ops;

  for (auto compute_op : func.getOps<ComputeOp>()) {
    if (compute_op.getCallee() == "AliasOp") {
      auto in_val = compute_op.getOperand(0);
      int in_idx = byre_alias.getOrCreateIndex(in_val);

      int leader_idx = byre_alias.leaderToIndex.getLeaderValue(in_idx);
      if (leader_idx != in_idx) {
        auto leader_val = byre_alias.values[leader_idx];
        // override operand and attr
        auto out_val = compute_op.getOperand(1);
        int out_idx = byre_alias.getOrCreateIndex(out_val);
        auto offset = byre_alias.offsets[out_idx];
        compute_op.setOperand(0, leader_val);
        compute_op->setAttr(
            "offset", IntegerAttr::get(IntegerType::get(ctx, 32), offset));

        if (leader_val.getDefiningOp() == nullptr) {
          compute_op->setAttr("arg_alias", UnitAttr::get(ctx));
        }
      }
    }
  }

  for (auto compute_op : func.getOps<ComputeOp>()) {
    if (compute_op.getCallee() == "AliasOp") {
      auto in_val = compute_op.getOperand(0);
      auto out_val = compute_op.getOperand(1);
      if (in_val.getType() == out_val.getType() &&
          compute_op->getAttrOfType<IntegerAttr>("offset").getInt() == 0) {
        out_val.replaceAllUsesExcept(in_val, compute_op);
      }
    }
  }

  for (auto op : func.getOps<ComputeOp>()) {
    if (op.callee() == "AliasOp") {
      auto value = op->getOperand(1);
      if (value.hasOneUse()) {
        remove_ops.emplace_back(op);
      }
    }
  };

  for (auto op : remove_ops) {
    op->erase();
  }
}

struct ByreHoldPass : public ByreFoldBase<ByreHoldPass> {

  ByreHoldPass() : ByreFoldBase() {}

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    foldAlias(func);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createByreFoldPass() {
  return std::make_unique<ByreHoldPass>();
}
