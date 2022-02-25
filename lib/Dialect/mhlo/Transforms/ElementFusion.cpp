//===- ElementFusion.cpp --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/ElementFusion.h"
#include "PassDetail.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Dialect/mhlo/Transforms/FusionUtil.h"
#include "byteir/Utils/IRRewrite.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

bool IsMhlo(Operation *op) {
  Dialect *dialect = op->getDialect();
  return dialect && isa<MhloDialect>(dialect);
}

bool IsValidSingleElemwiseOp(Operation* op) {
  return IsMhlo(op) &&
         (op->hasTrait<::mlir::OpTrait::Elementwise>() ||
          op->hasTrait<mhlo::OpTrait::BroadcastingElementwise>());
}

bool IsFusibleCandidate(Operation *op) {
  return IsMhlo(op) &&
         (op->hasTrait<::mlir::OpTrait::Elementwise>() ||
          op->hasTrait<mhlo::OpTrait::BroadcastingElementwise>() ||
          IsMhloConstantLike(op) || 
          isa<mhlo::BroadcastInDimOp, 
              mhlo::BroadcastOp, 
              mhlo::ReshapeOp>(op));
}

bool IsFusibleStart(Operation *op) {
  if (!IsMhlo(op)) return false;

  if (op->hasTrait<::mlir::OpTrait::Elementwise>() ||
      op->hasTrait<mhlo::OpTrait::BroadcastingElementwise>() ||
      isa<mhlo::ReshapeOp>(op)) {
    return true;
  }

  if (isa<mhlo::BroadcastInDimOp, 
          mhlo::BroadcastOp>(op)) {
    auto val = op->getOperand(0);
    auto def_op = val.getDefiningOp();
    return def_op && IsMhloConstantLike(def_op);
  }

  return false;
}

bool IsFusibleWith(Operation *target, Operation * /*start*/) {
  return IsMhlo(target) &&
         (target->hasTrait<::mlir::OpTrait::Elementwise>() ||
          target->hasTrait<mhlo::OpTrait::BroadcastingElementwise>() ||
          IsMhloConstantLike(target) || 
          isa<mhlo::BroadcastInDimOp, 
              mhlo::BroadcastOp, 
              mhlo::ReshapeOp>(target));
}

bool Replicate(Operation *op) { 
  return IsMhloConstantLike(op);
}

struct ElementFusionPass : public ElementFusionBase<ElementFusionPass> {

  ElementFusionPass(bool clusterSingleElemwiseOp) : ElementFusionBase() {
    this->clusterSingleElemwiseOp = clusterSingleElemwiseOp;
  }
  void runOnOperation() override {
    FuncOp funcOp = getOperation();

    for (auto &block : funcOp.getBlocks()) {
      ReplicateDefiningOp(&block, Replicate);
    }

    ProducerFusionPlanner planner(funcOp, IsFusibleCandidate, IsFusibleStart,
                                  IsFusibleWith);

    planner.Run();

    const MhloFusionPlan &plan = planner.GetFusionPlan();

    for (auto it = plan.rbegin(); it != plan.rend(); ++it) {
      auto &pattern = *it;
      if (pattern.size() > 1) {
        applyMhloFusionPattern(pattern, getByteIRElementwiseFusionAttrName());
      } else if (clusterSingleElemwiseOp.getValue()) {
        if (pattern.size() == 1 && IsValidSingleElemwiseOp(pattern[0])) {
          applyMhloFusionPattern(pattern, getByteIRElementwiseFusionAttrName());
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createElementFusionPass(bool clusterSingleElemwiseOp) {
  return std::make_unique<ElementFusionPass>(clusterSingleElemwiseOp);
}
