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
#include "mlir/IR/Builders.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include <functional>

using namespace byteir;
using namespace llvm;
using namespace mlir;
using namespace mlir::byre;

namespace {

struct ByreAliasAnalysis : public AliasAnalysis {
  ByreAliasAnalysis(mlir::Block *b, llvm::ArrayRef<mlir::Value> initial_copy,
                    std::function<bool(mlir::Operation &op)> is_alias)
      : AliasAnalysis(b, initial_copy, is_alias) {
    offsets.resize(values.size(), 0);
  }

  int GetOrCreateIndex(mlir::Value val) override {
    if (value_to_index.count(val) == 0) {
      int count = values.size();
      value_to_index[val] = count;
      values.push_back(val);
      leader_to_index.insert(count);
      offsets.push_back(0);
    }
    return value_to_index[val];
  }

  void RunOnBlock() override {
    if (block->empty())
      return;

    for (auto &op : block->without_terminator()) {
      if (is_alias(op)) {
        int in_idx = GetOrCreateIndex(op.getOperand(0));
        int in_leader = leader_to_index.getLeaderValue(in_idx);
        int out_idx = GetOrCreateIndex(op.getOperand(1));
        int out_leader = leader_to_index.getLeaderValue(out_idx);

        if (in_leader <= out_idx) {
          leader_to_index.unionSets(in_leader, out_leader);
        } else {
          leader_to_index.unionSets(out_leader, in_leader);
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

bool IsAliasOp(Operation &op) {
  if (auto compute_op = dyn_cast<byre::ComputeOp>(op)) {
    return compute_op.getCallee() == "AliasOp";
  }
  return false;
};

void FoldAlias(FuncOp func) {
  auto ctx = func.getContext();
  // use all args as initial_copy for alias
  SmallVector<Value> initial_copy;
  for (auto val : func.getArguments()) {
    initial_copy.push_back(val);
  }

  auto &func_block = func.getBody().front();
  ByreAliasAnalysis byre_alias(&func_block, initial_copy, IsAliasOp);
  byre_alias.RunOnBlock();

  SmallVector<ComputeOp> remove_ops;

  for (auto compute_op : func.getOps<ComputeOp>()) {
    if (compute_op.getCallee() == "AliasOp") {
      auto in_val = compute_op.getOperand(0);
      int in_idx = byre_alias.GetOrCreateIndex(in_val);

      int leader_idx = byre_alias.leader_to_index.getLeaderValue(in_idx);
      if (leader_idx != in_idx) {
        auto leader_val = byre_alias.values[leader_idx];
        // override operand and attr
        auto out_val = compute_op.getOperand(1);
        int out_idx = byre_alias.GetOrCreateIndex(out_val);
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
}

struct ByreHoldPass : public ByreFoldBase<ByreHoldPass> {

  ByreHoldPass() : ByreFoldBase() {}

  void runOnOperation() override {
    FuncOp func = getOperation();
    FoldAlias(func);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createByreFoldPass() {
  return std::make_unique<ByreHoldPass>();
}
