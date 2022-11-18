//===- ConvForwardFusion.cpp ----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"

#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

struct FuseConvBiasActPattern : public OpRewritePattern<ace::ActivateOp> {
  using OpRewritePattern<ace::ActivateOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ace::ActivateOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    mhlo::AddOp addOp =
        dyn_cast_or_null<mhlo::AddOp>(op.getInput().getDefiningOp());
    if (!addOp) {
      return failure();
    }
    mhlo::BroadcastInDimOp broadcastOp =
        dyn_cast_or_null<mhlo::BroadcastInDimOp>(
            addOp.getRhs().getDefiningOp());
    if (!broadcastOp || broadcastOp.getBroadcastDimensions().size() != 1) {
      return failure();
    }
    int64_t broadcastDim =
        (*broadcastOp.getBroadcastDimensions().begin()).getSExtValue();
    mhlo::ConvolutionOp convOp =
        dyn_cast_or_null<mhlo::ConvolutionOp>(addOp.getLhs().getDefiningOp());
    if (!convOp) {
      return failure();
    }

    SmallVector<Value> inputs{convOp.getLhs(), convOp.getRhs(),
                              broadcastOp.getOperand()};
    SmallVector<Value> outputs{op.getResult()};
    MhloFusionPattern pattern{convOp, broadcastOp, addOp, op};

    NamedAttrList originAttrs;
    handleConvAttribute(originAttrs, convOp, rewriter);

    NamedAttrList attrs;
    for (const auto &attr : originAttrs) {
      // check bias_add
      if (attr.getName() == "output_layout") {
        auto layout = attr.getValue().cast<StringAttr>().getValue();
        if (layout == "NCHW" && broadcastDim != 1) {
          return failure();
        }
        if (layout == "NHWC" && broadcastDim != 3) {
          return failure();
        }
      }

      byre::appendByreComputeAttr(attrs, attr.getName(), attr.getValue());
    }
    byre::appendByreComputeAttr(attrs, "act_func", op.getActFuncAttr());
    attrs.append(byre::getByreComputeName(),
                 rewriter.getStringAttr("ConvBiasOp"));

    mhlo::FusionOp fusionOp =
        createMhloFusionFromPattern(rewriter, inputs, outputs, pattern);
    fusionOp->setAttrs(attrs.getDictionary(getContext()));

    return success();
  }
};

struct FuseConvBiasPattern : public OpRewritePattern<mhlo::AddOp> {
  using OpRewritePattern<mhlo::AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    mhlo::BroadcastInDimOp broadcastOp =
        op.getRhs().getDefiningOp<mhlo::BroadcastInDimOp>();
    if (!broadcastOp || broadcastOp.getBroadcastDimensions().size() != 1) {
      return failure();
    }
    int64_t broadcastDim =
        (*broadcastOp.getBroadcastDimensions().begin()).getSExtValue();
    mhlo::ConvolutionOp convOp =
        op.getLhs().getDefiningOp<mhlo::ConvolutionOp>();
    if (!convOp) {
      return failure();
    }

    SmallVector<Value> inputs{convOp.getLhs(), convOp.getRhs(),
                              broadcastOp.getOperand()};
    SmallVector<Value> outputs{op.getResult()};
    MhloFusionPattern pattern{convOp, broadcastOp, op};

    NamedAttrList originAttrs;
    handleConvAttribute(originAttrs, convOp, rewriter);

    NamedAttrList attrs;
    for (const auto &attr : originAttrs) {
      // check bias_add
      if (attr.getName() == "output_layout") {
        auto layout = attr.getValue().cast<StringAttr>().getValue();
        if (layout == "NCHW" && broadcastDim != 1) {
          return failure();
        }
        if (layout == "NHWC" && broadcastDim != 3) {
          return failure();
        }
      }

      byre::appendByreComputeAttr(attrs, attr.getName(), attr.getValue());
    }
    byre::appendByreComputeAttr(attrs, "act_func",
                                rewriter.getStringAttr("none"));
    attrs.append(byre::getByreComputeName(),
                 rewriter.getStringAttr("ConvBiasOp"));

    mhlo::FusionOp fusionOp =
        createMhloFusionFromPattern(rewriter, inputs, outputs, pattern);
    fusionOp->setAttrs(attrs.getDictionary(getContext()));

    return success();
  }
};

struct ConvForwardFusionPass
    : public ConvForwardFusionBase<ConvForwardFusionPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateFuseConvForwardPatterns(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateFuseConvForwardPatterns(RewritePatternSet &patterns) {
  patterns.add<FuseConvBiasActPattern>(patterns.getContext(), 10);
  patterns.add<FuseConvBiasPattern>(patterns.getContext(), 5);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvForwardFusionPass() {
  return std::make_unique<ConvForwardFusionPass>();
}
