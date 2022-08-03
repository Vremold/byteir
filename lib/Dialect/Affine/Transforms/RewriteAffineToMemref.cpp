//===- RewriteAffineToMemref.cpp --------------------------------------*--- C++
//-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Affine/Transforms/RewriteAffineToMemref.h"
#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::memref;

namespace {

class ConvertLoad : public OpRewritePattern<mlir::AffineLoadOp> {
public:
  using OpRewritePattern<mlir::AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build vector.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getMemRef(),
                                                *resultOperands);
    return success();
  }
};

class ConvertStore : public OpRewritePattern<AffineStoreOp> {
public:
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    // Build memref.store valueToStore, memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
    return success();
  }
};

// a local popluate
void populateAffineLoadStoreConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertLoad, ConvertStore>(patterns.getContext());
}

struct RewriteAffineToMemrefPass
    : public RewriteAffineToMemrefBase<RewriteAffineToMemrefPass> {
public:
  RewriteAffineToMemrefPass() = default;
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    auto &ctx = getContext();
    ConversionTarget target(ctx);

    target.addLegalDialect<cf::ControlFlowDialect, memref::MemRefDialect>();

    target.addIllegalOp<mlir::AffineLoadOp, mlir::AffineStoreOp>();

    RewritePatternSet patterns(&ctx);
    populateAffineLoadStoreConversionPatterns(patterns);

    if (failed(applyPartialConversion(f, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createRewriteAffineToMemrefPass() {
  return std::make_unique<RewriteAffineToMemrefPass>();
}
