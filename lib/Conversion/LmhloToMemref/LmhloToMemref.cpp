//===- LmhloToMemref.cpp --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/LmhloToMemref/LmhloToMemref.h"
#include "../PassDetail.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::lmhlo;
using namespace mlir::memref;

namespace {

int64_t prod(ArrayRef<int64_t> a) {
  int64_t ret = 1;
  for (size_t i = 0; i < a.size(); ++i)
    ret *= a[i];
  return ret;
}

struct ConvertReshape : public OpRewritePattern<lmhlo::ReshapeOp> {
  using OpRewritePattern<lmhlo::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(lmhlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // handles static shape only
    auto allocOp = op.output().getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return failure();
    auto inMemRefType = op.operand().getType().cast<MemRefType>();
    auto outMemRefType = op.output().getType().cast<MemRefType>();
    auto input_shape = inMemRefType.getShape();
    auto output_shape = outMemRefType.getShape();

    // check: product of output's shape must equal to operand's shape
    if (prod(input_shape) != prod(output_shape))
      return failure();

    // create meta memref of output shape
    SmallVector<int64_t> shape;
    shape.push_back(output_shape.size());
    auto shapeMetaMemRefType = MemRefType::get(shape, rewriter.getI64Type());
    auto shape_allocOp =
        rewriter.create<memref::AllocOp>(op.getLoc(), shapeMetaMemRefType);
    auto const_op = rewriter.create<lmhlo::ConstOp>(
        op.getLoc(),
        GetI64ElementsAttr(output_shape, output_shape.size(), &rewriter),
        shape_allocOp.getResult());

    auto newMemRefType = MemRefType::get(
        outMemRefType.getShape(), outMemRefType.getElementType(),
        outMemRefType.getLayout(), inMemRefType.getMemorySpace());
    auto newReshapeOp = rewriter.create<memref::ReshapeOp>(
        op.getLoc(), newMemRefType, op.operand(), const_op.output());
    rewriter.replaceOp(allocOp, newReshapeOp.getResult());
    rewriter.eraseOp(op);

    return success();
  }
};

struct SliceToSubview : public OpRewritePattern<lmhlo::SliceOp> {
  using OpRewritePattern<lmhlo::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(lmhlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
    auto allocOp = op.output().getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return failure();
    auto inMemRefType = op.operand().getType().cast<MemRefType>();
    auto start_indices = SmallVector<int64_t>();
    auto limit_indices = SmallVector<int64_t>();
    auto strides = SmallVector<int64_t>();
    getValuesFromDenseIntElementsAttr(op.start_indices(), start_indices);
    getValuesFromDenseIntElementsAttr(op.limit_indices(), limit_indices);
    getValuesFromDenseIntElementsAttr(op.strides(), strides);
    auto input_shape = inMemRefType.getShape();

    if (start_indices.size() != limit_indices.size() ||
        limit_indices.size() != strides.size())
      return failure();
    // check: 0 <= start_indices[d] < limit_indices[d] < full_dim[d]
    // check: (limit_indices[d] - start_indices[d]) % strides[d] == 0
    for (size_t i = 0; i < start_indices.size(); ++i) {
      if (!(0 <= start_indices[i] && start_indices[i] < limit_indices[i] &&
            limit_indices[i] <= input_shape[i]))
        return failure();
      if ((limit_indices[i] - start_indices[i]) % strides[i] > 0)
        return failure();
    }

    SmallVector<int64_t> sizes;
    for (size_t i = 0; i < start_indices.size(); ++i)
      sizes.push_back((limit_indices[i] - start_indices[i]) / strides[i]);

    auto newSubViewOp = rewriter.create<memref::SubViewOp>(
        op.getLoc(), op.operand(), ArrayRef<int64_t>(start_indices),
        ArrayRef<int64_t>(sizes), ArrayRef<int64_t>(strides));
    rewriter.replaceOp(allocOp, newSubViewOp.getResult());
    rewriter.eraseOp(op);

    return success();
  }
};

struct LmhloToMemrefPass : public LmhloToMemrefBase<LmhloToMemrefPass> {
public:
  LmhloToMemrefPass() = default;
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    auto funcOp = getOperation();

    populateLmhloToMemrefPattern(patterns);
    target.addIllegalOp<lmhlo::ReshapeOp>();
    target.addIllegalOp<lmhlo::SliceOp>();
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError("LmhloToMemrefPass applyPatternsAndFoldGreedily "
                       "does not converge");
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateLmhloToMemrefPattern(RewritePatternSet &patterns) {
  patterns.add<ConvertReshape, SliceToSubview>(patterns.getContext());
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createLmhloToMemrefPass() {
  return std::make_unique<LmhloToMemrefPass>();
}
