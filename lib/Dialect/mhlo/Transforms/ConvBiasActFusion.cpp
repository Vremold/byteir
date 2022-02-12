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
#include "byteir/Dialect/mhlo/Util/Util.h"
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
    int64_t broadcast_dim =
        (*broadcastOp.broadcast_dimensions().begin()).getSExtValue();
    mhlo::ConvOp convOp =
        dyn_cast_or_null<mhlo::ConvOp>(addOp.lhs().getDefiningOp());
    if (!convOp) {
      return failure();
    }

    NamedAttrList origin_attrs;
    HandleConvAttribute(origin_attrs, convOp, rewriter);

    NamedAttrList attrs;
    for (const auto &attr : origin_attrs) {
      // check bias_add
      if (attr.getName() == "output_layout") {
        auto layout = attr.getValue().cast<StringAttr>().getValue();
        if (layout == "NCHW" && broadcast_dim != 1) {
          return failure();
        }
        if (layout == "NHWC" && broadcast_dim != 3) {
          return failure();
        }
      }

      byre::appendByreComputeAttr(attrs, attr.getName(), attr.getValue());
    }
    byre::appendByreComputeAttr(attrs, "act_func", op.act_funcAttr());
    attrs.append(byre::getByreComputeName(),
                 rewriter.getStringAttr("ConvBiasOp"));

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