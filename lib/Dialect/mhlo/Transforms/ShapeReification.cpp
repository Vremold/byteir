//===- ShapeReification.cpp -----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/ShapeReification.h"
#include "./PassDetail.h"
#include "byteir/Dialect/mhlo/ReifyShapes/Register.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace llvm;

namespace {

LogicalResult reifyShapes(OpBuilder &builder, Operation *op,
                          SmallVectorImpl<Value> &reifications) {
  if (!op)
    return failure();
  // TODO: support nested function call
  if (auto origin = dyn_cast<InferShapedTypeOpInterface>(op)) {
    if (failed(origin.reifyReturnTypeShapes(builder, origin->getOperands(),
                                            reifications))) {
      return failure();
    }
  } else if (auto reifyFunc =
                 reifyReturnTypeShapes(op->getName().getStringRef())) {
    if (failed(reifyFunc(op, builder, op->getOperands(), reifications))) {
      return failure();
    }
  } else if (auto customCall = dyn_cast<mhlo::CustomCallOp>(op)) {
    auto inferFunc = reifyReturnTypeShapes(customCall.call_target_name());
    if (!inferFunc) {
      return failure();
    }
    if (failed(inferFunc(op, builder, op->getOperands(), reifications)))
      return failure();
  } else {
    // Return failure if op doesn't have InferShapedTypeOpInterface and not
    // registered.
    return failure();
  }

  return success();
}

struct ShapeReificationOnTensorDimPattern
    : public OpRewritePattern<tensor::DimOp> {
  explicit ShapeReificationOnTensorDimPattern(MLIRContext *ctx)
      : OpRewritePattern<tensor::DimOp>(ctx) {
    // Recursively reify until we hit an op that doesn't support it.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const override {
    auto origin = op.source().getDefiningOp();
    SmallVector<Value, 1> reifications;

    if (failed(reifyShapes(rewriter, origin, reifications))) {
      return failure();
    }

    Value shape = reifications[op.source().cast<OpResult>().getResultNumber()];
    Value dimOfShape =
        rewriter.create<tensor::ExtractOp>(op.getLoc(), shape, op.index());

    // Insert cast, if needed.
    if (dimOfShape.getType() != op.getType()) {
      dimOfShape = rewriter.create<tensor::CastOp>(op.getLoc(), op.getType(),
                                                   dimOfShape);
    }

    rewriter.replaceOp(op, dimOfShape);
    return success();
  }
};

struct ShapeReificationPattern : public OpRewritePattern<shape::ShapeOfOp> {
  explicit ShapeReificationPattern(MLIRContext *ctx)
      : OpRewritePattern<shape::ShapeOfOp>(ctx) {
    // Recursively reify until we hit an op that doesn't support it.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(shape::ShapeOfOp op,
                                PatternRewriter &rewriter) const override {
    Operation *defOp = op.getArg().getDefiningOp();
    SmallVector<Value, 1> reifications;
    if (failed(reifyShapes(rewriter, defOp, reifications))) {
      return failure();
    }

    Value shape = reifications[op.getArg().cast<OpResult>().getResultNumber()];
    // Insert cast, if needed.
    if (shape.getType() != op.getType()) {
      shape = rewriter.create<tensor::CastOp>(op.getLoc(), op.getType(), shape);
    }

    rewriter.replaceOp(op, shape);
    return success();
  }
};

void PopulateShapeReificationPatterns(MLIRContext *ctx,
                                      RewritePatternSet *patterns) {
  patterns->add<ShapeReificationPattern, ShapeReificationOnTensorDimPattern>(
      ctx);
}

struct ShapeReificationPass
    : public ShapeReificationBase<ShapeReificationPass> {

  ShapeReificationPass()
      : ShapeReificationBase<ShapeReificationPass>::ShapeReificationBase() {
    // ReifyReturnType implementation could also be registered outside
    // ShapeReificationPass
    registerAllMhloReifyReturnTypeShapes();
  }

  void runOnOperation() override {
    // Collect patterns.
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    PopulateShapeReificationPatterns(ctx, &patterns);

    // Apply patterns from the bottom up. This ensures to need no more than one
    // iteration.
    GreedyRewriteConfig cfg;
    cfg.useTopDownTraversal = false;
    FuncOp f = getOperation();
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns), cfg))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createShapeReificationPass() {
  return std::make_unique<ShapeReificationPass>();
}