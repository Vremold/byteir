//===- ShapeReification.h -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/ShapeReification.h"
#include "./PassDetail.h"
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

// The function signature is similar to reifyReturnTypeShapes's, except that
// it has an additional argument of type `Operation *`. It should be easy if
// we decice to contribute some of the implementation to upstream later.
LogicalResult
reifyDotReturnTypeShapes(Operation *op, OpBuilder &builder, ValueRange operands,
                         SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) {
  auto dotOp = cast<mhlo::DotOp>(op);
  auto lhs_type = dotOp.lhs().getType().dyn_cast<ShapedType>();
  auto rhs_type = dotOp.rhs().getType().dyn_cast<ShapedType>();
  if (!lhs_type || !rhs_type || !lhs_type.hasRank() || !rhs_type.hasRank()) {
    return failure();
  }

  mhlo::DotOp::Adaptor adaptor(operands);
  auto lhs = adaptor.lhs();
  auto rhs = adaptor.rhs();
  SmallVector<Value> dimensions;

  // vector dot vector
  if (1 == lhs_type.getRank() && 1 == rhs_type.getRank()) {
    ;
  }
  // matrix dot vector
  else if (2 == lhs_type.getRank() && 1 == rhs_type.getRank()) {
    dimensions.push_back(builder.create<tensor::DimOp>(dotOp.getLoc(), lhs, 0));
  }
  // vector dot matrix
  else if (1 == lhs_type.getRank() && 2 == rhs_type.getRank()) {
    dimensions.push_back(builder.create<tensor::DimOp>(dotOp.getLoc(), rhs, 1));
  }
  // matrix dot matrix
  else if (2 == lhs_type.getRank() && 2 == rhs_type.getRank()) {
    dimensions.push_back(builder.create<tensor::DimOp>(dotOp.getLoc(), lhs, 0));
    dimensions.push_back(builder.create<tensor::DimOp>(dotOp.getLoc(), rhs, 1));
  } else {
    return failure();
  }
  reifiedReturnShapes.push_back(
      builder.create<tensor::FromElementsOp>(dotOp.getLoc(), dimensions));
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
    auto origin = op.source().getDefiningOp<InferShapedTypeOpInterface>();
    if (!origin)
      return failure();
    SmallVector<Value, 1> reifications;
    if (failed(origin.reifyReturnTypeShapes(rewriter, origin->getOperands(),
                                            reifications))) {
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
    if (!defOp)
      return failure();
    SmallVector<Value, 1> reifications;
    // TODO: support nested function call
    if (auto origin = dyn_cast<InferShapedTypeOpInterface>(defOp)) {
      if (failed(origin.reifyReturnTypeShapes(rewriter, origin->getOperands(),
                                              reifications))) {
        return failure();
      }
    } else if (isa<mhlo::DotOp>(defOp)) {
      if (failed(reifyDotReturnTypeShapes(defOp, rewriter, defOp->getOperands(),
                                          reifications))) {
        return failure();
      }
    } else {
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
