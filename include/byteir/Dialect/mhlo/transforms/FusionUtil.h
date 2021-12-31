//===- FusionUtil.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_MHLO_TRANSFORM_FUSIONUTIL_H
#define BYTEIR_MHLO_TRANSFORM_FUSIONUTIL_H

#include "byteir/Analysis/OpDependence.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <memory>


namespace mlir {

using MhloFusionPattern = llvm::SmallVector<Operation*, 8>;
using MhloFusionPlan = llvm::SmallVector<MhloFusionPattern, 8>;

// A generic way rewriting MhloFusionPattern to FusionOps
// This is from Mhlo repo.
// TODO push back to upstream if passible
void ApplyMhloFusionPattern(const MhloFusionPattern& pattern, llvm::StringRef tag);

// A predefined FusionPlanner using 3 criteria funcs
// It can be used to as a skeleton for fusion pass
// It checks only first degree producer only
// Producer means it only extends fusion along producer direction
// First degree means it only consider 1 op along fusion direction

// TODO implement consumer direction later
class ProducerFusionPlanner {
public:
  ProducerFusionPlanner(mlir::FuncOp funcOp,
    std::function<bool(Operation*)> fuse_candidate,
    std::function<bool(Operation*)> fuse_start,
    std::function<bool(Operation*, Operation*)> fuse_with);

  void Run();

  const MhloFusionPlan& GetFusionPlan() {
    return fusion_plan_;
  }

private:
  // OpDependenceInfo analysis
  std::unique_ptr<OpDependenceInfo> dependence_;

  // Fusible criteria functions
  std::function<bool(Operation*)> fuse_candidate_;

  std::function<bool(Operation*)> fuse_start_;

  std::function<bool(Operation*, Operation*)> fuse_with_;

  // a list of all candidates
  SmallVector<Operation*, 8> op_list_;

  // NodeMap from op to id
  llvm::DenseMap<Operation*, int> op_to_node_id_;

  // a UnionFind set
  llvm::EquivalenceClasses<int> leader_to_nodes_;

  // leader to value cnt, where 1 input counts +1, 1 output counts -1
  llvm::SmallDenseMap<int, llvm::DenseMap<Value, int>> leader_to_value_count_;

  // Fusion Plan
  MhloFusionPlan fusion_plan_;

  // return true, when two clusters already fused
  bool AlreayFused(Operation* pre_op, Operation* cur_op);

  // return true, when two clusters can be fused
  bool CheckFusionLegal(Operation* pre_op, Operation* cur_op);

  // fuse two clusters, each having each op
  void Merge(Operation* pre_op, Operation* cur_op);

};

} // namespace mlir

#endif // BYTEIR_MHLO_TRANSFORM_FUSIONUTIL_H
