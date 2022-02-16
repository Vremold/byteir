//===- HloFolder.cpp ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/HloFolder.h"
#include "PassDetail.h"
#include "byteir/Dialect/mhlo/Analysis/DimFromBroadcast.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace llvm;
using namespace byteir;

namespace {

static LogicalResult
AddScatterAddMatchAndRewriteHelper(mhlo::AddOp add_op, int idx,
                                   PatternRewriter &rewriter) {

  // Match
  mhlo::ScatterOp scatter_op =
      add_op.getOperand(idx).getDefiningOp<mhlo::ScatterOp>();

  if (!scatter_op) {
    return failure();
  }

  // check wthether scatter supported
  Region &region = scatter_op.update_computation();
  // only support single block
  if (region.getBlocks().size() != 1) {
    return failure();
  }

  auto &block = region.front();
  if (!IsBlockSingleAdd(&block)) {
    return failure();
  }

  Value initial_val = scatter_op.operand();
  if (!IsSplatMhloConstantValue(initial_val, (int64_t)0) &&
      !IsSplatMhloConstantValue(initial_val, 0.0)) {
    return failure();
  }

  // Rewrite
  int another_idx = 1 - idx;
  auto cloned = rewriter.clone(*scatter_op.getOperation());
  cloned->setOperand(0, add_op.getOperand(another_idx));
  rewriter.replaceOp(add_op, cloned->getResult(0));
  return success();
}

// Add + Scatter {add} -> Scatter
// TODO other scatter support
struct AddScatterAddToScatterPattern : public OpRewritePattern<mhlo::AddOp> {
  using OpRewritePattern<mhlo::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AddOp op,
                                PatternRewriter &rewriter) const override {

    // handle left
    if (failed(AddScatterAddMatchAndRewriteHelper(op, 0, rewriter))) {
      // handle right
      return AddScatterAddMatchAndRewriteHelper(op, 1, rewriter);
    }

    return success();
  }
};

struct RemoveTrivialTorchIndexSelect
    : public OpRewritePattern<mhlo::TorchIndexSelectOp> {
  using OpRewritePattern<mhlo::TorchIndexSelectOp>::OpRewritePattern;
  RemoveTrivialTorchIndexSelect(MLIRContext *context, DimFlagAnalysis *analysis)
      : OpRewritePattern(context), analysis_(analysis) {}

  LogicalResult matchAndRewrite(mhlo::TorchIndexSelectOp op,
                                PatternRewriter &rewriter) const override {
    uint64_t dim = op.dim();
    uint64_t batch_dims = op.batch_dims();
    Value index = op.index();
    Value input = op.input();

    auto index_shaped_type = index.getType().dyn_cast<ShapedType>();
    auto input_shaped_type = input.getType().dyn_cast<ShapedType>();
    if (batch_dims > 0 || index_shaped_type.getRank() > 1 ||
        !index_shaped_type || !index_shaped_type.hasStaticShape() ||
        !input_shaped_type || !input_shaped_type.hasStaticShape() ||
        index_shaped_type.getShape()[0] != input_shaped_type.getShape()[dim]) {
      return failure();
    }

    SmallVector<bool> from_broadcast = analysis_->GetDimFlag(input);
    if (!(int64_t(from_broadcast.size()) == input_shaped_type.getRank()) ||
        !from_broadcast[dim]) {
      return failure();
    }
    rewriter.replaceOp(op, input);
    return success();
  }

  DimFlagAnalysis *analysis_;
};

struct HloFolderPass : public HloFolderBase<HloFolderPass> {
  void runOnOperation() override {
    DimFromBroadcast dim_from_broadcast;
    DimFlagAnalysis dim_from_broadcast_analysis(&dim_from_broadcast);
    FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateHloFoldPatterns(patterns);
    patterns.add<RemoveTrivialTorchIndexSelect>(patterns.getContext(),
                                                &dim_from_broadcast_analysis);
    LogicalResult status =
        applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    if (failed(status)) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateHloFoldPatterns(RewritePatternSet &patterns) {
  patterns.add<AddScatterAddToScatterPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createHloFolderPass() {
  return std::make_unique<HloFolderPass>();
}