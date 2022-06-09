//===- FusionUtil.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_MHLO_UTIL_FUSIONUTIL_H
#define BYTEIR_MHLO_UTIL_FUSIONUTIL_H

#include "byteir/Analysis/OpDependence.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <memory>

namespace mlir {
class OpBuilder;

namespace mhlo {
class FusionOp;
} // namespace mhlo

using MhloFusionPattern = llvm::SmallVector<Operation *, 8>;
using MhloFusionPlan = llvm::SmallVector<MhloFusionPattern, 8>;

// A generic way rewriting MhloFusionPattern to FusionOps
// This is from Mhlo repo.
// TODO push back to upstream if passible

// This versionwill automatically scan and generate input/output with the
// sequence of defOp in the fusion's basic block.
// Often used: when input/output sequence order or number has no constraint.
mhlo::FusionOp createMhloFusionFromPattern(OpBuilder &b,
                                           const MhloFusionPattern &pattern);

// This version with inputs/outputs will use explicit given inputs/outputs.
// Often used: 1) when requiring the input/output sequence order,
//             2) when creating arg/return for unused input/output.
mhlo::FusionOp createMhloFusionFromPattern(OpBuilder &b, ValueRange inputs,
                                           ValueRange outputs,
                                           const MhloFusionPattern &pattern);

void applyMhloFusionPattern(const MhloFusionPattern &pattern,
                            llvm::StringRef tag);

// A predefined FusionPlanner using 3 criteria funcs
// It can be used to as a skeleton for fusion pass
// It checks only first degree producer only
// Producer means it only extends fusion along producer direction
// First degree means it only consider 1 op along fusion direction

// TODO implement consumer direction later
class ProducerFusionPlanner {
public:
  ProducerFusionPlanner(
      mlir::FuncOp funcOp, std::function<bool(Operation *)> fuse_candidate,
      std::function<bool(Operation *)> fuse_start,
      std::function<bool(Operation *)> fuse_trigger,
      std::function<bool(Operation *, Operation *)> fuse_with);

  void run();

  const MhloFusionPlan &getFusionPlan() { return fusion_plan_; }

private:
  // OpDependenceInfo analysis
  std::unique_ptr<OpDependenceInfo> dependence_;

  // Fusible criteria functions

  // Return true, if op is fusible
  std::function<bool(Operation *)> fuse_candidate_;

  // Return true, if op can be a starting op of a fusion
  std::function<bool(Operation *)> fuse_start_;

  // Return true, if op will actively trigger a fusion action
  // Note if false, the op won't actively trigger a fusion action with others,
  // but the op might still be fused by others' trigger
  std::function<bool(Operation *)> fuse_trigger_;

  // Return true, if two ops can be fused
  std::function<bool(Operation *, Operation *)> fuse_with_;

  // a list of all candidates
  SmallVector<Operation *, 8> op_list_;

  // NodeMap from op to id
  llvm::DenseMap<Operation *, int> op_to_node_id_;

  // a UnionFind set
  llvm::EquivalenceClasses<int> leader_to_nodes_;

  llvm::SmallDenseSet<int> fused_leaders_;

  // leader to value cnt, where use cnt
  llvm::SmallDenseMap<int, llvm::DenseMap<Value, int>> leader_to_value_count_;

  // Fusion Plan
  MhloFusionPlan fusion_plan_;

  // return true, when two clusters already fused
  bool alreadyFused(Operation *pre_op, Operation *cur_op);

  // return true, when two clusters can be fused
  bool checkFusionLegal(Operation *pre_op, Operation *cur_op);

  // fuse two clusters, each having each op
  void merge(Operation *pre_op, Operation *cur_op);
};

} // namespace mlir

#endif // BYTEIR_MHLO_UTIL_FUSIONUTIL_H
