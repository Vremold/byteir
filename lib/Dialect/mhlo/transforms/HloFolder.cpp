//===- HloFolder.cpp ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/transforms/HloFolder.h"
#include "byteir/Utils/Utils.h"
#include "PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace llvm;

namespace {

// Slice + Slice -> Slice

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
    auto broadcast_dimensions = broadcast_op.broadcast_dimensions();
    if (broadcast_dimensions.size() == 0) {
      mhlo::BroadcastInDimOp new_broadcast_op =
          rewriter.create<mhlo::BroadcastInDimOp>(
              op->getLoc(), op.getResult().getType(), broadcast_op.operand(),
              rewriter.getI64TensorAttr({}));
      rewriter.replaceOp(op, new_broadcast_op.getResult());
      return success();
    } else if (broadcast_dimensions.size() == 1) {
      auto permutaion = op.permutation();
      int64_t index = 0;
      for (auto p : permutaion) {
        if (p == *broadcast_dimensions.begin()) {
          break;
        }
        index++;
      }
      mhlo::BroadcastInDimOp new_broadcast_op =
          rewriter.create<mhlo::BroadcastInDimOp>(
              op->getLoc(), op.getResult().getType(), broadcast_op.operand(),
              rewriter.getI64TensorAttr({index}));
      rewriter.replaceOp(op, new_broadcast_op.getResult());
      return success();
    } else {
      // TODO(liuyuanqiang): handle more dims in broadcast
      return failure();
    }
    return failure();
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

struct HloFolderPass : public HloFolderBase<HloFolderPass> {
  void runOnFunction() override;
};

} // namespace

void mlir::populateHloFoldPatterns(RewritePatternSet &patterns) {
  patterns.add(std::make_unique<BroadcastInDimTransposeToBroadcastInDimPattern>(
      patterns.getContext()));
  patterns.add(std::make_unique<TransposeTransposeToTransposePattern>(
      patterns.getContext()));
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