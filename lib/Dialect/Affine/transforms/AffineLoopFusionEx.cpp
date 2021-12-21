//===- AffineLoopFusionEx.cpp ---------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Affine/transforms/AffineLoopFusionEx.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/LoopFusionUtils.h"
#include "PassDetail.h"
#include <utility>

using namespace llvm;
using namespace mlir;

namespace {

// TODO: maybe move this to util if it is useful for others
Operation* leastDominantDefiningOp(
  Operation* first,
  Operation* op, 
  DominanceInfo& domInfo) {

  Operation* cur_pos = first;
  for (auto val : op->getOperands()) {
    if (val.getDefiningOp() == nullptr) {
      continue;
    }

    if (domInfo.properlyDominates(cur_pos, val.getDefiningOp())) {
      cur_pos = val.getDefiningOp();
    }
  }

  return cur_pos;
}

bool IsHoistableOp(Operation* op) {
  // everything but dealloc
  return isa<memref::AllocOp>(op) ||
    isa<memref::CollapseShapeOp>(op) ||
    isa<memref::DimOp>(op) ||
    isa<memref::ExpandShapeOp>(op) ||
    isa<memref::ReshapeOp>(op);
}

void HoistNonAffineLoop(
  FuncOp funcOp, DominanceInfo& domInfo, 
  SmallVector<AffineForOp>& loop_collector) {

  SmallVector<std::pair<Operation*, Operation*>> movable_ops;
  for (auto& block : funcOp.getBody()) {
    auto& first = block.front();
    for (auto& op : block.without_terminator()) {
      // skip AffineFor
      if (auto forOp = dyn_cast<AffineForOp>(op)) {
        loop_collector.push_back(forOp);
        continue;
      }

      // skip non-hoistable op
      if (!IsHoistableOp(&op)) continue;

      auto pos = leastDominantDefiningOp(&first, &op, domInfo);
      if (pos != &op) {
        movable_ops.emplace_back(&op, pos);
      }
    }
  }

  // real movement happens here
  for (auto& p : movable_ops) {
    p.first->moveAfter(p.second);
  }
}

// This is a temp fix for affine fusion 
void UpdateComputationSliceState(mlir::ComputationSliceState& sliceUnion, MLIRContext* ctx) {
  sliceUnion.lbs[0] = AffineMap::getMultiDimIdentityMap(1, ctx);
  // generate d0 + 1
  AffineExpr d0 = getAffineDimExpr(0, ctx);
  AffineExpr out = d0 + 1;
  SmallVector<AffineExpr, 4> result;
  result.push_back(out);
  sliceUnion.ubs[0] = AffineMap::get(1, 0, result, ctx);
}

void FuseAffineLoopEx(FuncOp funcOp, ArrayRef<AffineForOp> loops) {
  // early return if only 1 or 0 loop
  if (loops.size() <= 1) return;

  auto first = loops[0];
  for (size_t i = 1; i < loops.size(); ++i) {
    AffineForOp forOp = loops[i];
    ComputationSliceState sliceUnion;
    FusionResult result = canFuseLoops(forOp, first, 1, &sliceUnion);

    if (result.value == FusionResult::Success) {
      // FIXME: mlir's fuseLoops seems buggy in some cases
      // just fix sliceUnion to single-step loop (lb = d0, ub = d0+1) so it can trigger fusion
      // TODO change it back after it is fixed.
      UpdateComputationSliceState(sliceUnion, funcOp.getContext());
      fuseLoops(forOp, first, sliceUnion);
      forOp.erase();
    }
  }
}

struct AffineLoopFusionExPass : public AffineLoopFusionExBase<AffineLoopFusionExPass> {
  AffineLoopFusionExPass(const std::string& anchor) 
    : AffineLoopFusionExBase() {
    anchorTag = anchor;
  }
  void runOnFunction() override;
};

} // namespace anonymous

void AffineLoopFusionExPass::runOnFunction() {
  auto funcOp = getFunction();

  if (!anchorTag.empty() && !funcOp->hasAttrOfType<UnitAttr>(anchorTag)) return;


  auto& domInfo = getAnalysis<DominanceInfo>();

  SmallVector<AffineForOp> loop_collector;

  HoistNonAffineLoop(funcOp, domInfo, loop_collector);

  FuseAffineLoopEx(funcOp, loop_collector);
}

std::unique_ptr<FunctionPass>
mlir::createAffineLoopFusionExPass(const std::string& anchor) {
  return std::make_unique<AffineLoopFusionExPass>(anchor);
}
