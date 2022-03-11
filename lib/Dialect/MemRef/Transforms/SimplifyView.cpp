//===- SimplifyView.cpp ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/MemRef/Transforms/SimplifyView.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/Transforms/ComposeSubView.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;

// some code from mlir's ComposeSubView
namespace {

// Util linearize
Optional<int64_t> getLinearizeOffset(ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes) {
  if (sizes.size() == 0) return 0;

  int64_t sum = 0;
  int64_t prod = 1;
  int rank = offsets.size();
  bool prevDynamic = false;
  for (int i = rank - 1; i >= 0; --i) {
    if (prevDynamic || offsets[i] == MemRefType::getDynamicStrideOrOffset()) {
      return llvm::None;
    }

    sum += offsets[i] * prod;

    if (sizes[i] == ShapedType::kDynamicSize) {
      prevDynamic = true;
    }
    prod *= sizes[i];
  }

  return sum;
}

inline int64_t getBytes(int64_t bits) { 
  return (bits + 7) >> 3; 
}

struct ComposeSubViewOfView : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {

    auto srcView = op.source().getDefiningOp<memref::ViewOp>();
    if (!srcView) return failure();


    // A 'SubViewOp' can be "rank-reducing" by eliminating dimensions of the
    // output memref that are statically known to be equal to 1. We do not
    // allow 'sourceOp' to be a rank-reducing subview because then our two
    // 'SubViewOp's would have different numbers of offset/size/stride
    // parameters (just difficult to deal with, not impossible if we end up
    // needing it).
    if (srcView.getType().getRank() != op.getType().getRank()) {
      return failure();
    }

    // only support ContiguousRowMajor
    if (!isStaticShapeAndContiguousRowMajor(op.getType())) {
      return failure();
    }


    SmallVector<OpFoldResult> strides = op.getMixedStrides();

    // Because we only support input strides of 1, the output stride is also
    // always 1.
    if (llvm::all_of(strides, [](OpFoldResult &valueOrAttr) {
          Attribute attr = valueOrAttr.dyn_cast<Attribute>();
          return attr && attr.cast<IntegerAttr>().getInt() == 1;
        })) {
    } else {
      return failure();
    }

    // FIXME: only support constant for now
    SmallVector<int64_t> offsets;
    for (auto opOffset : op.getMixedOffsets()) {
      // We only support static offset.
      if (opOffset.is<Value>()) {
        return failure();
      }
      Attribute opOffsetAttr = opOffset.dyn_cast<Attribute>();
      offsets.push_back(opOffsetAttr.cast<IntegerAttr>().getInt());
    }

    auto maybeInt = getLinearizeOffset(offsets, srcView.getType().getShape());
    if (!maybeInt.hasValue()) failure();

    int64_t offsetInByte 
      = maybeInt.getValue() * getBytes(op.getType().getElementTypeBitWidth());

    auto lieanrizedOffset = 
      rewriter.create<arith::ConstantIndexOp>(op.getLoc(), offsetInByte);

    auto newShift = rewriter.create<arith::AddIOp>(
        op.getLoc(), srcView.byte_shift(), lieanrizedOffset.getResult());

    // New MemRefType
    auto newMemRefType = MemRefType::get(
      op.getType().getShape(), 
      op.getType().getElementType(), 
      MemRefLayoutAttrInterface{}, 
      op.getType().getMemorySpace());

    rewriter.replaceOpWithNewOp<memref::ViewOp>(
      op, newMemRefType, srcView.source(), newShift.getResult(), op.sizes());

    return success();
  }
};

struct ComposeViewOfView : public OpRewritePattern<memref::ViewOp> {
  using OpRewritePattern<memref::ViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ViewOp op,
                                PatternRewriter &rewriter) const override {
    auto srcView = op.source().getDefiningOp<memref::ViewOp>();
    if (!srcView) return failure();

    auto newShift = rewriter.create<arith::AddIOp>(
      op.getLoc(), op.byte_shift(), srcView.byte_shift());

    rewriter.replaceOpWithNewOp<memref::ViewOp>(op, 
      op.getType(), srcView.source(), newShift.getResult(), op.sizes());

    return success();
  }
};

struct SimplifyViewPass : public SimplifyViewBase<SimplifyViewPass> {
public:
  SimplifyViewPass() = default;
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    OwningRewritePatternList patterns(funcOp.getContext());
    populateSimplifyViewPattern(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError("SimplifyViewPass applyPatternsAndFoldGreedily "
                       "does not converge");
      signalPassFailure();
    }
  }
};

} // namespace anonymous

void mlir::populateSimplifyViewPattern(RewritePatternSet &patterns) {
  populateComposeSubViewPatterns(patterns, patterns.getContext());
  patterns.add<ComposeViewOfView, 
               ComposeSubViewOfView>(patterns.getContext());
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createSimplifyViewPass() {
  return std::make_unique<SimplifyViewPass>();
}
