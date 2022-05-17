//===- LmhloToLace.cpp ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/LmhloToLace/LmhloToLace.h"
#include "../PassDetail.h"
#include "byteir/Dialect/Lace/LaceDialect.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace llvm;
using namespace mlir;

namespace {
// replace memref.alloc + lmhlo.reshape with lace.reshape
struct ConvertReshape : public OpConversionPattern<lmhlo::ReshapeOp> {
  using OpConversionPattern<lmhlo::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(lmhlo::ReshapeOp op, lmhlo::ReshapeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // handles static shape only
    auto allocOp = adaptor.output().getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return failure();

    auto newReshapeOp = rewriter.create<lace::ReshapeOp>(
        op.getLoc(), adaptor.output().getType(), adaptor.operand());
    rewriter.replaceOp(allocOp, newReshapeOp.getResult());
    rewriter.eraseOp(op);

    return success();
  }
};

// replace memref.alloc + lmhlo.slice with lace.slice
struct ConvertSlice : public OpConversionPattern<lmhlo::SliceOp> {
  using OpConversionPattern<lmhlo::SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(lmhlo::SliceOp op, lmhlo::SliceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto allocOp = adaptor.output().getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return failure();

    SmallVector<int64_t> startIndices, limitIndices, strides;
    getValuesFromDenseIntElementsAttr(op.start_indices(), startIndices);
    getValuesFromDenseIntElementsAttr(op.limit_indices(), limitIndices);
    getValuesFromDenseIntElementsAttr(op.strides(), strides);

    auto srcMemRefType =
        adaptor.operand().getType().dyn_cast_or_null<MemRefType>();
    auto dstMemRefType =
        adaptor.output().getType().dyn_cast_or_null<MemRefType>();

    if (!srcMemRefType || !dstMemRefType)
      return failure();

    if (!lace::SliceOp::isValid(srcMemRefType, dstMemRefType, startIndices,
                                limitIndices, strides))
      return failure();

    auto newSliceOp = rewriter.create<lace::SliceOp>(
        op.getLoc(), adaptor.output().getType(), adaptor.operand(),
        op.start_indices(), op.limit_indices(), op.strides());
    rewriter.replaceOp(allocOp, newSliceOp.getResult());
    rewriter.eraseOp(op);

    return success();
  }
};

struct LmhloToLacePass : public LmhloToLaceBase<LmhloToLacePass> {
public:
  LmhloToLacePass() : LmhloToLaceBase() {}
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    auto funcOp = getOperation();

    populateLmhloToLacePattern(patterns);
    target.addIllegalOp<lmhlo::ReshapeOp>();
    target.addIllegalOp<lmhlo::SliceOp>();
    target.addLegalDialect<lace::LaceDialect>();
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateLmhloToLacePattern(RewritePatternSet &patterns) {
  patterns.add<ConvertReshape, ConvertSlice>(patterns.getContext());
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createLmhloToLacePass() {
  return std::make_unique<LmhloToLacePass>();
}
