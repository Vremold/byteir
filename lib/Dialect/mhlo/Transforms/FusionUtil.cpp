//===- FusionUtil.cpp -----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/FusionUtil.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::mhlo;
using namespace llvm;

namespace {

llvm::DenseMap<Value, int> InitValueCount(Operation *op) {
  llvm::DenseMap<Value, int> ret;

  // output
  for (auto val : op->getResults()) {
    ret[val] = UseCount(val);
  }

  // input
  for (auto val : op->getOperands()) {
    // skip block arg
    if (val.getDefiningOp() == nullptr) {
      continue;
    }
    ret[val]--;
  }

  return ret;
}

} // namespace

// This code is from mhlo repo
// but it was in the local namespace, so cannot be directly call.
// TODO: we might update upstream to make it accessible later
mhlo::FusionOp
mlir::createMhloFusionFromPattern(OpBuilder &b, ValueRange inputs,
                                  ValueRange outputs,
                                  const MhloFusionPattern &pattern) {

  b.setInsertionPoint(pattern.back());

  SmallVector<Location, 4> locations;
  locations.reserve(pattern.size());
  for (Operation *op : pattern) {
    locations.push_back(op->getLoc());
  }
  Location fused_loc = FusedLoc::get(pattern.back()->getContext(), locations);

  SmallVector<Type, 4> output_types;
  output_types.reserve(outputs.size());
  for (Value v : outputs) {
    output_types.push_back(v.getType());
  }

  SmallDenseSet<Operation *> fused_set(pattern.begin(), pattern.end());
  SmallDenseSet<Operation *> consumers_set;

  SmallVector<Operation *, 4> consumers_vec;
  auto first_iter = pattern.front()->getIterator();
  auto last_iter = pattern.back()->getIterator();

  for (Operation &cur_op : llvm::make_range(first_iter, last_iter)) {
    // isn't fused op && consumer's op
    // move this after fusion op
    if (!fused_set.contains(&cur_op)) {
      // fused op's consumer or consumer's consumer
      bool is_consumer = llvm::any_of(
          cur_op.getOperands(), [&fused_set, &consumers_set](Value v) {
            auto op = v.getDefiningOp();
            return fused_set.contains(op) || consumers_set.contains(op);
          });
      if (is_consumer) {
        consumers_set.insert(&cur_op);
        consumers_vec.push_back(&cur_op);
      }
    }
  }

  for (auto op : llvm::reverse(consumers_vec)) {
    op->moveAfter(pattern.back());
  }

  FusionOp fusion = b.create<mhlo::FusionOp>(fused_loc, output_types, inputs);
  Region &region = fusion.fused_computation();
  region.push_back(new Block);
  Block &block = region.front();

  for (Operation *op : pattern) {
    op->moveBefore(&block, block.end());
  }

  b.setInsertionPoint(&block, block.end());
  b.create<mhlo::ReturnOp>(fused_loc, outputs);

  for (auto output_and_result : llvm::zip(outputs, fusion.getResults())) {
    Value output = std::get<0>(output_and_result);
    Value fusion_result = std::get<1>(output_and_result);
    for (OpOperand &use : llvm::make_early_inc_range(output.getUses())) {
      if (use.getOwner()->getBlock() != &block)
        use.set(fusion_result);
    }
  }

  return fusion;
}

mhlo::FusionOp
mlir::createMhloFusionFromPattern(OpBuilder &b,
                                  const MhloFusionPattern &pattern) {
  SmallVector<Value, 4> inputs = GetInputsOfCluster(pattern);
  SmallVector<Value, 4> outputs = GetOutputsOfCluster(pattern);
  return createMhloFusionFromPattern(b, inputs, outputs, pattern);
}

void mlir::applyMhloFusionPattern(const MhloFusionPattern &pattern,
                                  StringRef attachTag) {
  OpBuilder b(pattern.back());
  auto fusion = createMhloFusionFromPattern(b, pattern);
  if (!attachTag.empty()) {
    fusion->setAttr(attachTag, UnitAttr::get(fusion.getContext()));
  }
}

mlir::ProducerFusionPlanner::ProducerFusionPlanner(
    FuncOp funcOp, std::function<bool(Operation *)> is_fusible,
    std::function<bool(Operation *)> fuse_start,
    std::function<bool(Operation *, Operation *)> fuse_with)
    : fuse_candidate_(is_fusible), fuse_start_(fuse_start),
      fuse_with_(fuse_with) {

  // if empty function jus terminate
  if (funcOp.getBlocks().empty()) {
    return;
  }

  Block &entry_block = funcOp.getBlocks().front();

  dependence_ = std::make_unique<OpDependenceInfo>(&entry_block);

  for (Operation &op : entry_block) {
    // skip non-fusible
    if (!fuse_candidate_(&op)) {
      continue;
    }

    int idx = op_list_.size();
    op_list_.push_back(&op);
    op_to_node_id_[&op] = idx;
    leader_to_nodes_.insert(idx);
    leader_to_value_count_[idx] = InitValueCount(&op);
  }
}

bool mlir::ProducerFusionPlanner::AlreayFused(Operation *pre_op,
                                              Operation *cur_op) {
  assert(op_to_node_id_.count(pre_op) > 0);
  assert(op_to_node_id_.count(cur_op) > 0);

  int pre_id = op_to_node_id_[pre_op];
  int cur_id = op_to_node_id_[cur_op];
  return leader_to_nodes_.isEquivalent(pre_id, cur_id);
}

bool mlir::ProducerFusionPlanner::CheckFusionLegal(Operation *pre_op,
                                                   Operation *cur_op) {
  assert(op_to_node_id_.count(pre_op) > 0);
  assert(op_to_node_id_.count(cur_op) > 0);

  int pre_leader = leader_to_nodes_.getLeaderValue(op_to_node_id_[pre_op]);
  const auto &pre_cluster = leader_to_value_count_[pre_leader];
  int cur_id = op_to_node_id_[cur_op];

  for (auto it : pre_cluster) {

    // skip input
    if (it.second <= 0) {
      continue;
    }

    // output's use
    for (auto &use : it.first.getUses()) {
      auto another_user = use.getOwner();

      // skip if another user is curOp
      if (another_user == cur_op)
        continue;

      // check if another user is a candidate
      if (op_to_node_id_.count(another_user) > 0) {
        auto another_id = op_to_node_id_[another_user];

        // skip if another user already fused with curOp
        // or already fused wiht pre_op
        if (leader_to_nodes_.isEquivalent(cur_id, another_id) ||
            leader_to_nodes_.isEquivalent(pre_leader, another_id)) {
          continue;
        }
      }

      // check if there is another path going through another user to curOp
      // if so, return false
      if (dependence_->properlyDepends(another_user, cur_op)) {
        return false;
      }
    }
  }

  return true;
}

void mlir::ProducerFusionPlanner::Merge(Operation *pre_op, Operation *cur_op) {
  assert(op_to_node_id_.count(pre_op) > 0);
  assert(op_to_node_id_.count(cur_op) > 0);

  int pre_leader = leader_to_nodes_.getLeaderValue(op_to_node_id_[pre_op]);
  int cur_leader = leader_to_nodes_.getLeaderValue(op_to_node_id_[cur_op]);

  int small_leader = pre_leader < cur_leader ? pre_leader : cur_leader;
  int large_leader = pre_leader < cur_leader ? cur_leader : pre_leader;

  leader_to_nodes_.unionSets(small_leader, large_leader);
  // keep small one
  auto &small_value_cnt = leader_to_value_count_[small_leader];
  auto &large_value_cnt = leader_to_value_count_[large_leader];
  for (auto it : large_value_cnt) {

    // merge two use cnt
    small_value_cnt[it.first] += it.second;

    if (small_value_cnt[it.first] == 0) {
      small_value_cnt.erase(it.first);
    }
  }

  leader_to_value_count_.erase(large_leader);
}

void mlir::ProducerFusionPlanner::Run() {

  SmallVector<Operation *, 8> op_iteration = op_list_;

  for (auto *op : op_iteration) {

    // fusion only can start when fuse_start_ is true
    if (!fuse_start_(op)) {
      continue;
    }

    // check fusion in the operand sequence
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto val = op->getOperand(i);
      auto op_def = val.getDefiningOp();

      // skip block arg (input args)
      // or not in candidate
      // or already fused
      if (op_def == nullptr || op_to_node_id_.count(op_def) == 0 ||
          AlreayFused(op_def, op)) {
        continue;
      }

      if (!fuse_with_(op_def, op) || !CheckFusionLegal(op_def, op)) {
        continue;
      }

      // now we can fuse
      Merge(op_def, op);
    }
  }

  llvm::SmallDenseMap<int, int> leader_to_offset;

  for (auto it = leader_to_nodes_.begin(); it != leader_to_nodes_.end(); ++it) {
    auto id = it->getData();
    auto *op = op_list_[id];
    auto leader = leader_to_nodes_.getLeaderValue(id);

    if (leader_to_offset.count(leader) == 0) {
      leader_to_offset[leader] = fusion_plan_.size();
      MhloFusionPattern pattern;
      pattern.push_back(op);
      fusion_plan_.push_back(pattern);
    } else {
      int offset = leader_to_offset[leader];
      fusion_plan_[offset].push_back(op);
    }
  }
}
