//===- HloFolder.cpp ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/transforms/HloFolder.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "PassDetail.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace llvm;

namespace {

// TODO(liuyuanqiang): push these patterns to upstream

// BroadcastInDim + Transpose -> BroadcastInDim
struct BroadcastInDimTransposeToBroadcastInDimPattern
    : public OpRewritePattern<mhlo::TransposeOp> {
  using OpRewritePattern<mhlo::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    mhlo::BroadcastInDimOp broadcast_op =
        dyn_cast_or_null<mhlo::BroadcastInDimOp>(op.operand().getDefiningOp());
    if (!broadcast_op) {
      return failure();
    }
    SmallVector<int64_t> new_broadcast_dimensions;
    auto broadcast_dimensions = broadcast_op.broadcast_dimensions();
    auto permutation = op.permutation();
    for (auto dimension : broadcast_dimensions) {
      int64_t index = 0;
      for (auto p : permutation) {
        if (p == dimension) {
          new_broadcast_dimensions.push_back(index);
          break;
        }
        index++;
      }
    }
    mhlo::BroadcastInDimOp new_broadcast_op =
        rewriter.create<mhlo::BroadcastInDimOp>(
            op->getLoc(), op.getResult().getType(), broadcast_op.operand(),
            rewriter.getI64TensorAttr(new_broadcast_dimensions));
    rewriter.replaceOp(op, new_broadcast_op.getResult());
    return success();
  }
};

// Transpose + Transpose -> Transpose
struct TransposeTransposeToTransposePattern
    : public OpRewritePattern<mhlo::TransposeOp> {
  using OpRewritePattern<mhlo::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    mhlo::TransposeOp transpose_op =
        op.operand().getDefiningOp<mhlo::TransposeOp>();
    if (!transpose_op) {
      return failure();
    }
    SmallVector<int64_t> permutation, permutation1;
    getValuesFromDenseIntElementsAttr(op.permutation(), permutation);
    getValuesFromDenseIntElementsAttr(transpose_op.permutation(), permutation1);
    for (size_t i = 0; i < permutation.size(); i++) {
      permutation[i] = permutation1[permutation[i]];
    }
    auto loc = rewriter.getFusedLoc({op->getLoc(), transpose_op->getLoc()});
    mhlo::TransposeOp new_transpose_op = rewriter.create<mhlo::TransposeOp>(
        loc, op.getResult().getType(), transpose_op.operand(),
        rewriter.getI64TensorAttr(permutation));
    rewriter.replaceOp(op, new_transpose_op.getResult());
    return success();
  }
};

static LogicalResult AddScatterAddMatchAndRewriteHelper(
  mhlo::AddOp add_op, int idx,
  PatternRewriter& rewriter) {

  // Match 
  mhlo::ScatterOp scatter_op =
    add_op.getOperand(idx).getDefiningOp<mhlo::ScatterOp>();

  if (!scatter_op) {
    return failure();
  }

  // check wthether scatter supported
  Region& region = scatter_op.update_computation();
  // only support single block
  if (region.getBlocks().size() != 1) {
    return failure();
  }

  auto& block = region.front();
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
struct AddScatterAddToScatterPattern
  : public OpRewritePattern<mhlo::AddOp> {
  using OpRewritePattern<mhlo::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AddOp op,
    PatternRewriter& rewriter) const override {

    // handle left
    if (failed(AddScatterAddMatchAndRewriteHelper(op, 0, rewriter))) {
      // handle right
      return AddScatterAddMatchAndRewriteHelper(op, 1, rewriter);
    }
 
    return success();
  }
};

struct HloFolderPass : public HloFolderBase<HloFolderPass> {
  void runOnFunction() override;
};

} // namespace

void mlir::populateHloFoldPatterns(RewritePatternSet &patterns) {
  patterns.add<
    BroadcastInDimTransposeToBroadcastInDimPattern, 
    TransposeTransposeToTransposePattern,
    AddScatterAddToScatterPattern
  >(patterns.getContext());
}

void HloFolderPass::runOnFunction() {
  FuncOp funcOp = getFunction();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateHloFoldPatterns(patterns);
  LogicalResult status =
      applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  if (failed(status)) {
    signalPassFailure();
  }
}

std::unique_ptr<FunctionPass> mlir::createHloFolderPass() {
  return std::make_unique<HloFolderPass>();
}