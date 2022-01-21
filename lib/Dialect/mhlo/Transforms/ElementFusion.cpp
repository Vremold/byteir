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

bool IsFusibleCandidate(Operation *op) {
  // FIXME (LWC) Tentatively disable constant fusion to avoid a bug
  return IsMhlo(op) &&
         (op->hasTrait<::mlir::OpTrait::Elementwise>() ||
          IsSplatMhloConstant(op) || isa<mhlo::BroadcastInDimOp>(op) ||
          isa<mhlo::BroadcastOp>(op) || isa<mhlo::ReshapeOp>(op) ||
          isa<mhlo::ClampOp>(op) || isa<mhlo::SelectOp>(op));
}

bool IsFusibleStart(Operation *op) {
  return IsMhlo(op) && (op->hasTrait<::mlir::OpTrait::Elementwise>() ||
                        isa<mhlo::ReshapeOp>(op) || isa<mhlo::ClampOp>(op) ||
                        isa<mhlo::SelectOp>(op));
  //&& !isa<mhlo::ShiftRightLogicalOp>(op);
}

bool IsFusibleWith(Operation *target, Operation * /*start*/) {
  // FIXME (LWC) Tentatively disable constant fusion to avoid a bug
  return IsMhlo(target) &&
         (target->hasTrait<::mlir::OpTrait::Elementwise>() ||
          IsSplatMhloConstant(target) || isa<mhlo::BroadcastInDimOp>(target) ||
          isa<mhlo::BroadcastOp>(target) || isa<mhlo::ReshapeOp>(target) ||
          isa<mhlo::ClampOp>(target) || isa<mhlo::SelectOp>(target));
  //&& !isa<mhlo::ShiftRightLogicalOp>(target);
}

struct ElementFusionPass : public ElementFusionBase<ElementFusionPass> {

  ElementFusionPass(const std::string &tag) : ElementFusionBase() {
    attachTag = tag;
  }
  void runOnOperation() override {
    FuncOp funcOp = getOperation();

    for (auto &block : funcOp.getBlocks()) {
      ReplicateDefiningOp(&block, IsSplatMhloConstant);
    }

    ProducerFusionPlanner planner(funcOp, IsFusibleCandidate, IsFusibleStart,
                                  IsFusibleWith);

    planner.Run();

    const MhloFusionPlan &plan = planner.GetFusionPlan();

    for (auto it = plan.rbegin(); it != plan.rend(); ++it) {
      auto &pattern = *it;
      if (pattern.size() > 1) {
        ApplyMhloFusionPattern(pattern, attachTag);
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createElementFusionPass(const std::string &attachTag) {
  return std::make_unique<ElementFusionPass>(attachTag);
}
