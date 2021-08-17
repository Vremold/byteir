//===- MhloPreprocessing.cpp ----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/transforms/TransposeFolder.h"
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

struct TransposeFolderPass : public TransposeFolderBase<TransposeFolderPass> {
  void runOnFunction() override;
};

} // namespace

void mlir::populateFoldTransposePatterns(RewritePatternSet &patterns) {
  patterns.add(std::make_unique<BroadcastInDimTransposeToBroadcastInDimPattern>(
      patterns.getContext()));
}

void TransposeFolderPass::runOnFunction() {
  FuncOp funcOp = getFunction();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateFoldTransposePatterns(patterns);
  LogicalResult status =
      applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  if (failed(status)) {
    signalPassFailure();
  }
}

std::unique_ptr<FunctionPass> mlir::createTransposeFolderPass() {
  return std::make_unique<TransposeFolderPass>();
}