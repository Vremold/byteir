//===- ConvBiasActFusion.cpp ----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/ConvBiasActFusion.h"
#include "PassDetail.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/Common.h"
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

// return 'true' for correctly broadcast
bool handleConvLayout(NamedAttrList &attrs,
                      mhlo::ConvDimensionNumbersAttr dimension_numbers,
                      int64_t broadcast_dim, PatternRewriter &rewriter) {
  StringAttr input_layout;
  auto input_batch_dimension = dimension_numbers.getInputBatchDimension();
  auto input_feature_dimension = dimension_numbers.getInputFeatureDimension();
  assert(dimension_numbers.getInputSpatialDimensions().size() == 2);
  if (input_batch_dimension == 0 && input_feature_dimension == 1) {
    input_layout = rewriter.getStringAttr("NCHW");
    if (broadcast_dim != 1) {
      return false;
    }
  } else if (input_batch_dimension == 0 && input_feature_dimension == 3) {
    input_layout = rewriter.getStringAttr("NHWC");
    if (broadcast_dim != 3) {
      return false;
    }
  } else {
    assert(false && "Unsupported convolution input layout.");
  }

  StringAttr output_layout;
  auto output_batch_dimension = dimension_numbers.getOutputBatchDimension();
  auto output_feature_dimension = dimension_numbers.getOutputFeatureDimension();
  assert(dimension_numbers.getOutputSpatialDimensions().size() == 2);
  if (output_batch_dimension == 0 && output_feature_dimension == 1) {
    output_layout = rewriter.getStringAttr("NCHW");
  } else if (output_batch_dimension == 0 && output_feature_dimension == 3) {
    output_layout = rewriter.getStringAttr("NHWC");
  } else {
    assert(false && "Unsupported convolution output layout.");
  }

  assert(input_layout.getValue() == output_layout.getValue() &&
         "Input layout should be same as output layout.");

  StringAttr kernel_layout;
  auto kernel_input_feature_dimension =
      dimension_numbers.getKernelInputFeatureDimension();
  auto kernel_output_feature_dimension =
      dimension_numbers.getKernelOutputFeatureDimension();
  assert(dimension_numbers.getKernelSpatialDimensions().size() == 2);
  if (kernel_input_feature_dimension == 1 &&
      kernel_output_feature_dimension == 0) {
    kernel_layout = rewriter.getStringAttr("KCRS");
  } else if (kernel_input_feature_dimension == 2 &&
             kernel_output_feature_dimension == 3) {
    kernel_layout = rewriter.getStringAttr("RSCK");
  } else {
    assert(false && "Unsupported convolution kernel layout.");
  }

  byre::appendByreComputeAttr(attrs, "input_layout", input_layout);
  byre::appendByreComputeAttr(attrs, "output_layout", output_layout);
  byre::appendByreComputeAttr(attrs, "kernel_layout", kernel_layout);
  return true;
}

struct FuseConvBiasActPattern : public OpRewritePattern<ace::ActivateOp> {
  using OpRewritePattern<ace::ActivateOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ace::ActivateOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    mhlo::AddOp addOp =
        dyn_cast_or_null<mhlo::AddOp>(op.input().getDefiningOp());
    if (!addOp) {
      return failure();
    }
    mhlo::BroadcastInDimOp broadcastOp =
        dyn_cast_or_null<mhlo::BroadcastInDimOp>(addOp.rhs().getDefiningOp());
    if (!broadcastOp || broadcastOp.broadcast_dimensions().size() != 1) {
      return failure();
    }
    mhlo::ConvOp convOp =
        dyn_cast_or_null<mhlo::ConvOp>(addOp.lhs().getDefiningOp());
    if (!convOp) {
      return failure();
    }

    NamedAttrList attrs;
    attrs.append(byre::getByreComputeName(),
                 rewriter.getStringAttr("ConvBiasOp"));
    if (!handleConvLayout(
            attrs, convOp.dimension_numbers(),
            (*broadcastOp.broadcast_dimensions().begin()).getSExtValue(),
            rewriter)) {
      return failure();
    }
    SmallVector<int64_t> window_strides{1, 1};
    if (convOp.window_strides()) {
      window_strides.clear();
      getValuesFromDenseIntElementsAttr(convOp.window_stridesAttr(),
                                        window_strides);
    }
    SmallVector<int64_t> padding{0, 0, 0, 0};
    if (convOp.padding()) {
      padding.clear();
      getValuesFromDenseIntElementsAttr(convOp.paddingAttr(), padding);
    }
    SmallVector<int64_t> lhs_dilation{1, 1};
    if (convOp.lhs_dilation()) {
      lhs_dilation.clear();
      getValuesFromDenseIntElementsAttr(convOp.lhs_dilationAttr(),
                                        lhs_dilation);
    }
    SmallVector<int64_t> rhs_dilation{1, 1};
    if (convOp.rhs_dilation()) {
      rhs_dilation.clear();
      getValuesFromDenseIntElementsAttr(convOp.rhs_dilationAttr(),
                                        rhs_dilation);
    }

    // TODO: window_reversal attribute
    byre::appendByreComputeAttr(attrs, "act_func", op.act_funcAttr());
    byre::appendByreComputeAttr(attrs, "window_strides",
                                rewriter.getI64ArrayAttr(window_strides));
    byre::appendByreComputeAttr(attrs, "padding",
                                rewriter.getI64ArrayAttr(padding));
    byre::appendByreComputeAttr(attrs, "lhs_dilation",
                                rewriter.getI64ArrayAttr(lhs_dilation));
    byre::appendByreComputeAttr(attrs, "rhs_dilation",
                                rewriter.getI64ArrayAttr(rhs_dilation));
    byre::appendByreComputeAttr(attrs, "feature_group_count",
                                convOp.feature_group_countAttr());
    byre::appendByreComputeAttr(attrs, "batch_group_count",
                                convOp.batch_group_countAttr());

    Location loc =
        rewriter.getFusedLoc({op->getLoc(), addOp->getLoc(),
                              broadcastOp->getLoc(), convOp->getLoc()});
    mhlo::FusionOp fusionOp = rewriter.create<mhlo::FusionOp>(
        loc, op.getResult().getType(),
        ArrayRef<Value>{convOp.lhs(), convOp.rhs(), broadcastOp.operand()});
    op->replaceAllUsesWith(fusionOp.getResults());
    Region &region = fusionOp.fused_computation();
    Block &block = region.emplaceBlock();
    {
      // assume that there is no other reference of conv's result
      OpBuilder::InsertionGuard guard(rewriter);
      convOp->moveBefore(&block, block.end());
      broadcastOp->moveBefore(&block, block.end());
      addOp->moveBefore(&block, block.end());
      op->moveBefore(&block, block.end());

      rewriter.setInsertionPoint(&block, block.end());
      rewriter.create<mhlo::ReturnOp>(loc, op.getResult());
    }
    fusionOp->setAttrs(attrs.getDictionary(getContext()));

    return success();
  }
};

struct ConvBiasActFusionPass
    : public ConvBiasActFusionBase<ConvBiasActFusionPass> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateFuseConvBiasActPatterns(patterns);
    LogicalResult status =
        applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    if (failed(status)) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateFuseConvBiasActPatterns(RewritePatternSet &patterns) {
  patterns.add(std::make_unique<FuseConvBiasActPattern>(patterns.getContext()));
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createConvBiasActFusionPass() {
  return std::make_unique<ConvBiasActFusionPass>();
}