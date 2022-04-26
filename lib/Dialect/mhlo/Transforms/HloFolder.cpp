//===- HloFolder.cpp ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/HloFolder.h"
#include "PassDetail.h"
#include "byteir/Dialect/mhlo/Analysis/DimFromBroadcast.h"
#include "byteir/Dialect/mhlo/Transforms/CanonicalExt.h"
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
using namespace byteir;
using namespace mlir::mhlo;

namespace {

//===----------------------------------------------------------------------===//
// Add + Scatter => Scatter Pattern
//===----------------------------------------------------------------------===//

static LogicalResult
AddScatterAddMatchAndRewriteHelper(mhlo::AddOp add_op, int idx,
                                   PatternRewriter &rewriter) {

  // Match
  mhlo::ScatterOp scatter_op =
      add_op.getOperand(idx).getDefiningOp<mhlo::ScatterOp>();

  if (!scatter_op) {
    return failure();
  }

  // check wthether scatter supported
  Region &region = scatter_op.update_computation();
  // only support single block
  if (region.getBlocks().size() != 1) {
    return failure();
  }

  auto &block = region.front();
  if (!IsBlockSingleAdd(&block)) {
    return failure();
  }

  Value initial_val = scatter_op.operand();
  if (!IsSplatMhloConstantValue(initial_val, (int64_t)0) &&
      !IsSplatMhloConstantValue(initial_val, 0.0)) {
    return failure();
  }

  // Rewrite
  int another_idx = 1 - idx;
  auto cloned = rewriter.clone(*scatter_op.getOperation());
  cloned->setOperand(0, add_op.getOperand(another_idx));
  rewriter.replaceOp(add_op, cloned->getResult(0));
  return success();
}

// Add + Scatter {add} -> Scatter
// TODO other scatter support
struct AddScatterAddToScatterPattern : public OpRewritePattern<mhlo::AddOp> {
  using OpRewritePattern<mhlo::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AddOp op,
                                PatternRewriter &rewriter) const override {

    // handle left
    if (failed(AddScatterAddMatchAndRewriteHelper(op, 0, rewriter))) {
      // handle right
      return AddScatterAddMatchAndRewriteHelper(op, 1, rewriter);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// RemoveTrivialTorchIndexSelect Pattern
//===----------------------------------------------------------------------===//

struct RemoveTrivialTorchIndexSelect
    : public OpRewritePattern<mhlo::TorchIndexSelectOp> {
  using OpRewritePattern<mhlo::TorchIndexSelectOp>::OpRewritePattern;
  RemoveTrivialTorchIndexSelect(MLIRContext *context, DimFlagAnalysis *analysis)
      : OpRewritePattern(context), analysis_(analysis) {}

  LogicalResult matchAndRewrite(mhlo::TorchIndexSelectOp op,
                                PatternRewriter &rewriter) const override {
    uint64_t dim = op.dim();
    uint64_t batch_dims = op.batch_dims();
    Value index = op.index();
    Value input = op.input();

    auto index_shaped_type = index.getType().dyn_cast<ShapedType>();
    auto input_shaped_type = input.getType().dyn_cast<ShapedType>();
    if (batch_dims > 0 || index_shaped_type.getRank() > 1 ||
        !index_shaped_type || !index_shaped_type.hasStaticShape() ||
        !input_shaped_type || !input_shaped_type.hasStaticShape() ||
        index_shaped_type.getShape()[0] != input_shaped_type.getShape()[dim]) {
      return failure();
    }

    SmallVector<bool> from_broadcast = analysis_->GetDimFlag(input);
    if (!(int64_t(from_broadcast.size()) == input_shaped_type.getRank()) ||
        !from_broadcast[dim]) {
      return failure();
    }
    rewriter.replaceOp(op, input);
    return success();
  }

  DimFlagAnalysis *analysis_;
};

//===----------------------------------------------------------------------===//
// ConvFollowedByMulOrAdd Pattern
// TODO: handle similar cases of dot op followed by mul or add
//===----------------------------------------------------------------------===//

// Return the expanded constOp if applicable, return None if not. Applicable if
// all following constraint satisfied:
// 1. the op's input has static shape
// 2. op's input rank equals 1, or it is equal to output rank
// 3. there's at most one dim in input shape whose size is not equal to 1, and
//     it should be euqal to featureDim
// 4. the input's DefiningOp is of type mhlo::ConstOp
// 5. the const op's attr is of type DenseElementsAttr
Optional<ConstOp> getBroadcastedConstOp(BroadcastInDimOp op,
                                        int64_t featureDim) {
  Value broadInDimInput = op.operand();
  ShapedType broadInDimInpShape = broadInDimInput.getType().cast<ShapedType>();
  Value broadInDimOutput = op->getResult(0);
  ShapedType broadInDimOupShape = broadInDimOutput.getType().cast<ShapedType>();

  // Only need to check the input shape of broadcast_in_dim
  if (!broadInDimInpShape.hasStaticShape())
    return None;

  // op's input rank equals 1, or it is equal to output rank
  if (broadInDimInpShape.getRank() == 1) {
    SmallVector<int64_t> broadcastDims;
    int64_t bdim = (*op.broadcast_dimensions().begin()).getSExtValue();
    if (featureDim != bdim)
      return None;
  } else if (broadInDimInpShape.getRank() == broadInDimOupShape.getRank()) {
    int64_t nonOneDim = -1;
    for (int64_t i = 0; i < broadInDimInpShape.getRank(); ++i) {
      int64_t dimSize = broadInDimInpShape.getDimSize(i);
      if (dimSize != 1) {
        if (nonOneDim >= 0)
          return None;
        else {
          nonOneDim = i;
        }
      }
    }

    // There's at most one dim whose size is not equal to 1, and it should be
    // euqal to featureDim.
    if (nonOneDim != -1 && nonOneDim != featureDim)
      return None;
  } else {
    return None;
  }

  auto constOp = dyn_cast_or_null<ConstOp>(broadInDimInput.getDefiningOp());
  if (!constOp)
    return None;

  if (!constOp.value().dyn_cast_or_null<DenseElementsAttr>())
    return None;

  return constOp;
}

struct ConvFollowedByMulOrAdd : public OpRewritePattern<mhlo::ConvOp> {
  using OpRewritePattern<mhlo::ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConvOp convOp,
                                PatternRewriter &rewriter) const override {
    Value convOrBiasOut = convOp->getResult(0);
    if (!convOrBiasOut.hasOneUse())
      return failure();

    Operation *convOrBiasUser = *convOrBiasOut.user_begin();
    int64_t featureDim = convOp.dimension_numbers().getOutputFeatureDimension();
    Value convWeight = convOp.rhs();

    if (!convWeight.getDefiningOp() ||
        !isa<ConstOp>(convWeight.getDefiningOp()))
      return failure();
    if (!convWeight.hasOneUse())
      return failure();
    if (!convWeight.getType().cast<ShapedType>().hasStaticShape())
      return failure();

    ArrayRef<int64_t> convWeightShape =
        convWeight.getType().cast<ShapedType>().getShape();
    Type elemType = convWeight.getType().cast<ShapedType>().getElementType();

    // handle the conv + bias scenario
    auto biasAddOp = dyn_cast_or_null<mhlo::AddOp>(convOrBiasUser);
    ConstOp biasConst = nullptr;
    BroadcastInDimOp biasBroadcastInDimOp = nullptr;
    if (biasAddOp) {
      // Here we update `convOrBiasOut` and `convOrBiasUser`
      convOrBiasOut = biasAddOp->getResult(0);
      if (!convOrBiasOut.hasOneUse())
        return failure();
      convOrBiasUser = *convOrBiasOut.user_begin();

      unsigned convOperandNumber =
          convOp->getResult(0).use_begin()->getOperandNumber();
      assert(convOperandNumber < 2);
      auto broadInDimOp = dyn_cast_or_null<mhlo::BroadcastInDimOp>(
          biasAddOp->getOperand(1 - convOperandNumber).getDefiningOp());
      if (!broadInDimOp)
        return failure();

      auto maybeConstOp = getBroadcastedConstOp(broadInDimOp, featureDim);
      if (!maybeConstOp.hasValue())
        return failure();

      biasConst = maybeConstOp.getValue();
      biasBroadcastInDimOp = broadInDimOp;
    }

    unsigned convOrBiasOperandNumber =
        convOrBiasOut.use_begin()->getOperandNumber();

    if (auto scaleOp = dyn_cast_or_null<MulOp>(convOrBiasUser)) {
      auto broadInDimOp = dyn_cast_or_null<mhlo::BroadcastInDimOp>(
          scaleOp->getOperand(1 - convOrBiasOperandNumber).getDefiningOp());
      if (!broadInDimOp)
        return failure();

      auto maybeConstOp = getBroadcastedConstOp(broadInDimOp, featureDim);
      if (!maybeConstOp.hasValue())
        return failure();
      ConstOp constOp = maybeConstOp.getValue();

      // Start to construct a new subgraph which could be const folded.
      if (!constOp->isBeforeInBlock(convOp))
        constOp->moveBefore(convOp);

      // construct new conv weight
      OpBuilder builder(convOp);
      auto newReshapeType = RankedTensorType::get(
          {convOrBiasOut.getType().cast<ShapedType>().getDimSize(featureDim)},
          elemType);
      ReshapeOp newReshapeOp = builder.create<mhlo::ReshapeOp>(
          constOp->getLoc(), newReshapeType, constOp.output());
      auto newBroadInDimType = RankedTensorType::get(convWeightShape, elemType);
      auto broadAttr = DenseIntElementsAttr::get(
          RankedTensorType::get({1}, builder.getIntegerType(64)),
          {convOp.dimension_numbers().getKernelOutputFeatureDimension()});
      BroadcastInDimOp newBroadInDimOp = builder.create<mhlo::BroadcastInDimOp>(
          constOp->getLoc(), newBroadInDimType, newReshapeOp->getResult(0),
          broadAttr);
      MulOp newMulOp = builder.create<MulOp>(constOp->getLoc(), convWeight,
                                             newBroadInDimOp->getResult(0));
      convOp->setOperand(1, newMulOp->getResult(0));

      // construct new conv bias
      if (biasAddOp) {
        OpBuilder builder(biasAddOp);
        ReshapeOp newReshapeOp = builder.create<mhlo::ReshapeOp>(
            constOp->getLoc(), biasConst.output().getType(), constOp.output());
        MulOp newMulOp = builder.create<MulOp>(
            constOp->getLoc(), biasConst.output(), newReshapeOp->getResult(0));
        biasBroadcastInDimOp->setOperand(0, newMulOp->getResult(0));
      }

      // update conv's uses
      scaleOp->getResult(0).replaceAllUsesWith(convOrBiasOut);

    } else if (auto offsetOp = dyn_cast_or_null<AddOp>(convOrBiasUser)) {
      auto broadInDimOp = dyn_cast_or_null<mhlo::BroadcastInDimOp>(
          offsetOp->getOperand(1 - convOrBiasOperandNumber).getDefiningOp());
      if (!broadInDimOp)
        return failure();

      auto maybeConstOp = getBroadcastedConstOp(broadInDimOp, featureDim);
      if (!maybeConstOp.hasValue())
        return failure();
      ConstOp constOp = maybeConstOp.getValue();

      // Start to construct a new subgraph which could be const folded.
      if (!constOp->isBeforeInBlock(convOp))
        constOp->moveBefore(convOp);

      // construct new conv bias
      assert(biasAddOp);
      OpBuilder builder(biasAddOp);
      ReshapeOp newReshapeOp = builder.create<mhlo::ReshapeOp>(
          constOp->getLoc(), biasConst.output().getType(), constOp.output());
      AddOp newAddOp = builder.create<AddOp>(
          constOp->getLoc(), biasConst.output(), newReshapeOp->getResult(0));
      biasBroadcastInDimOp->setOperand(0, newAddOp->getResult(0));

      // update conv's uses
      offsetOp->getResult(0).replaceAllUsesWith(convOrBiasOut);

    } else {
      return failure();
    }

    return success();
  }
};

struct HloFolderPass : public HloFolderBase<HloFolderPass> {
  void runOnOperation() override {
    DimFromBroadcast dim_from_broadcast;
    DimFlagAnalysis dim_from_broadcast_analysis(&dim_from_broadcast);
    FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateHloFoldPatterns(patterns);
    patterns.add<RemoveTrivialTorchIndexSelect>(patterns.getContext(),
                                                &dim_from_broadcast_analysis);
    // also add canoncializationExt pattern
    mhlo::getCanonicalizationExtPatterns(patterns, patterns.getContext());

    LogicalResult status =
        applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    if (failed(status)) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateHloFoldPatterns(RewritePatternSet &patterns) {
  patterns.add<AddScatterAddToScatterPattern>(patterns.getContext());
  patterns.add<ConvFollowedByMulOrAdd>(patterns.getContext());
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createHloFolderPass() {
  return std::make_unique<HloFolderPass>();
}