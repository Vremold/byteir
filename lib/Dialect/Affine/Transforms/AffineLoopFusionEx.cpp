//===- AffineLoopFusionEx.cpp ---------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Affine/Transforms/AffineLoopFusionEx.h"
#include "PassDetail.h"
#include "byteir/Utils/Hoist.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include <utility>

using namespace llvm;
using namespace mlir;

namespace {

// TODO: maybe move this to util if it is useful for others
Operation *leastDominantDefiningOp(Operation *first, Operation *op,
                                   DominanceInfo &domInfo) {

  Operation *cur_pos = first;
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

bool IsHoistUpOp(Operation *op) {
  return isa<memref::AllocOp, memref::CollapseShapeOp, memref::DimOp,
             memref::ExpandShapeOp, memref::ReshapeOp>(op);
}

void collectAffineLopps(func::FuncOp funcOp,
                        SmallVector<AffineForOp> &loop_collector) {

  for (auto &block : funcOp.getBody()) {
    for (auto &op : block.without_terminator()) {
      // skip AffineFor
      if (auto forOp = dyn_cast<AffineForOp>(op)) {
        loop_collector.push_back(forOp);
        continue;
      }
    }
  }
}

// This is a temp fix for affine fusion
void UpdateComputationSliceState(mlir::ComputationSliceState &sliceUnion,
                                 MLIRContext *ctx) {
  sliceUnion.lbs[0] = AffineMap::getMultiDimIdentityMap(1, ctx);
  // generate d0 + 1
  AffineExpr d0 = getAffineDimExpr(0, ctx);
  AffineExpr out = d0 + 1;
  SmallVector<AffineExpr, 4> result;
  result.push_back(out);
  sliceUnion.ubs[0] = AffineMap::get(1, 0, result, ctx);
}

void fuseAffineLoopEx(func::FuncOp funcOp, ArrayRef<AffineForOp> loops) {
  // early return if only 1 or 0 loop
  if (loops.size() <= 1)
    return;

  auto first = loops[0];
  for (size_t i = 1; i < loops.size(); ++i) {
    AffineForOp forOp = loops[i];
    ComputationSliceState sliceUnion;
    FusionResult result = canFuseLoops(forOp, first, 1, &sliceUnion);

    if (result.value == FusionResult::Success) {
      // FIXME: mlir's fuseLoops seems buggy in some cases
      // just fix sliceUnion to single-step loop (lb = d0, ub = d0+1) so it can
      // trigger fusion
      // TODO change it back after it is fixed.
      UpdateComputationSliceState(sliceUnion, funcOp.getContext());
      fuseLoops(forOp, first, sliceUnion);
      forOp.erase();
    }
  }
}

struct AffineLoopFusionExPass
    : public AffineLoopFusionExBase<AffineLoopFusionExPass> {
  AffineLoopFusionExPass(const std::string &anchor) : AffineLoopFusionExBase() {
    anchorTag = anchor;
  }
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    if (!anchorTag.empty() && !funcOp->hasAttrOfType<UnitAttr>(anchorTag))
      return;

    auto &domInfo = getAnalysis<DominanceInfo>();

    SmallVector<AffineForOp> loopCollection;

    collectAffineLopps(funcOp, loopCollection);

    for (auto &block : funcOp.getBody()) {
      hoistUpOpsInBlock(&block, domInfo, IsHoistUpOp);
    }

    fuseAffineLoopEx(funcOp, loopCollection);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createAffineLoopFusionExPass(const std::string &anchor) {
  return std::make_unique<AffineLoopFusionExPass>(anchor);
}
