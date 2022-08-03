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
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  // only support one operand one update
  if (scatter_op.operands().size() != 1 || scatter_op.updates().size() != 1 ||
      scatter_op->getNumResults() != 1) {
    return failure();
  }

  auto &block = region.front();
  if (!isBlockSingleOp<mhlo::AddOp>(&block)) {
    return failure();
  }

  Value initial_val = scatter_op.operands()[0];
  if (!isSplatMhloConstantValue(initial_val, (int64_t)0) &&
      !isSplatMhloConstantValue(initial_val, 0.0)) {
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
  RemoveTrivialTorchIndexSelect(MLIRContext *context, DimFlagAnalysis *analysis)
      : OpRewritePattern<mhlo::TorchIndexSelectOp>(context),
        analysis_(analysis) {}

  LogicalResult matchAndRewrite(mhlo::TorchIndexSelectOp op,
                                PatternRewriter &rewriter) const override {
    uint64_t dim = op.dim();
    uint64_t batch_dims = op.batch_dims();
    Value index = op.index();
    Value input = op.operand();

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
// PadConvolution Pattern
//===----------------------------------------------------------------------===//

struct PadConvToConvPattern : public OpRewritePattern<mhlo::ConvolutionOp> {
  using OpRewritePattern<mhlo::ConvolutionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    auto padOp = op.lhs().getDefiningOp<mhlo::PadOp>();
    if (!padOp || !isZeroAttribute(padOp.interior_padding())) {
      return failure();
    }
    auto constOp = padOp.padding_value().getDefiningOp<mhlo::ConstantOp>();
    if (!constOp || !isZeroAttribute(constOp.value())) {
      return failure();
    }

    const auto edge_padding_low = padOp.edge_padding_low().getValues<int64_t>();
    const auto edge_padding_high =
        padOp.edge_padding_high().getValues<int64_t>();
    auto dimension_numbers = op.dimension_numbers();
    auto input_spatial_dims = dimension_numbers.getInputSpatialDimensions();
    llvm::SmallDenseSet<int64_t> input_spatial_dims_set(
        input_spatial_dims.begin(), input_spatial_dims.end());
    for (size_t i = 0; i < edge_padding_low.size(); i++) {
      if (!input_spatial_dims_set.contains(i)) {
        if (edge_padding_low[i] != 0 || edge_padding_high[i] != 0) {
          return failure();
        }
      }
    }

    SmallVector<int64_t> oldPadding(input_spatial_dims.size() * 2, 0);
    if (op.padding().hasValue()) {
      oldPadding =
          SmallVector<int64_t>(op.paddingAttr().getValues<int64_t>().begin(),
                               op.paddingAttr().getValues<int64_t>().end());
    }
    SmallVector<int64_t> newPadding;
    for (size_t i = 0; i < input_spatial_dims.size(); i++) {
      newPadding.push_back(edge_padding_low[input_spatial_dims[i]] +
                           oldPadding[i * 2]);
      newPadding.push_back(edge_padding_high[input_spatial_dims[i]] +
                           oldPadding[i * 2 + 1]);
    }
    auto newPaddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get(
            {static_cast<int64_t>(input_spatial_dims.size()), 2},
            rewriter.getI64Type()),
        newPadding);

    auto newOp = cast<mhlo::ConvolutionOp>(rewriter.clone(*op));
    newOp.setOperand(0, padOp.operand());
    newOp.paddingAttr(newPaddingAttr);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
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
// 4. the input's DefiningOp is of type mhlo::ConstantOp
// 5. the const op's attr is of type DenseElementsAttr
Optional<ConstantOp> getBroadcastedConstOp(BroadcastInDimOp op,
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

  auto constOp = dyn_cast_or_null<ConstantOp>(broadInDimInput.getDefiningOp());
  if (!constOp)
    return None;

  if (!constOp.value().dyn_cast_or_null<DenseElementsAttr>())
    return None;

  return constOp;
}

struct ConvOrConvBiasFollowedByBroadcastOp
    : public OpRewritePattern<mhlo::ConvolutionOp> {
  using OpRewritePattern<mhlo::ConvolutionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConvolutionOp convOp,
                                PatternRewriter &rewriter) const override {
    Value convOrBiasOut = convOp->getResult(0);
    if (!convOrBiasOut.hasOneUse())
      return failure();

    Operation *convOrBiasUser = *convOrBiasOut.user_begin();
    int64_t featureDim = convOp.dimension_numbers().getOutputFeatureDimension();
    Value convWeight = convOp.rhs();

    if (!convWeight.getDefiningOp() ||
        !isa<ConstantOp>(convWeight.getDefiningOp()))
      return failure();
    if (!convWeight.hasOneUse())
      return failure();
    if (!convWeight.getType().cast<ShapedType>().hasStaticShape())
      return failure();

    Type elemType = convWeight.getType().cast<ShapedType>().getElementType();

    // handle the conv + bias scenario
    auto biasAddOp = dyn_cast_or_null<mhlo::AddOp>(convOrBiasUser);
    ConstantOp biasConst = nullptr;
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
      auto broadInDimOp = biasAddOp->getOperand(1 - convOperandNumber)
                              .getDefiningOp<mhlo::BroadcastInDimOp>();
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
      auto broadInDimOp = scaleOp->getOperand(1 - convOrBiasOperandNumber)
                              .getDefiningOp<mhlo::BroadcastInDimOp>();
      if (!broadInDimOp)
        return failure();

      auto maybeConstOp = getBroadcastedConstOp(broadInDimOp, featureDim);
      if (!maybeConstOp.hasValue())
        return failure();
      ConstantOp constOp = maybeConstOp.getValue();

      // Start to construct a new subgraph which could be const folded.
      if (!constOp->isBeforeInBlock(convOp))
        constOp->moveBefore(convOp);

      // construct new conv weight
      OpBuilder builder(convOp);
      auto convWeightType = convOp.rhs().getType().cast<ShapedType>();
      auto weightFeatureDim =
          convOp.dimension_numbers().getKernelOutputFeatureDimension();
      ReshapeOp newReshapeOp = builder.create<mhlo::ReshapeOp>(
          constOp->getLoc(),
          RankedTensorType::get({convWeightType.getDimSize(weightFeatureDim)},
                                convWeightType.getElementType()),
          constOp.output());
      BroadcastInDimOp newBroadInDimOp = builder.create<mhlo::BroadcastInDimOp>(
          constOp->getLoc(), convWeightType, newReshapeOp->getResult(0),
          rewriter.getI64TensorAttr({weightFeatureDim}));
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
      auto broadInDimOp = offsetOp->getOperand(1 - convOrBiasOperandNumber)
                              .getDefiningOp<mhlo::BroadcastInDimOp>();
      if (!broadInDimOp)
        return failure();

      auto maybeConstOp = getBroadcastedConstOp(broadInDimOp, featureDim);
      if (!maybeConstOp.hasValue())
        return failure();
      ConstantOp constOp = maybeConstOp.getValue();

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

    } else if (auto subOp = dyn_cast_or_null<SubtractOp>(convOrBiasUser)) {
      // conv_or_bias - a => conv_or_bias + (- a)

      // b_const should be rhs
      auto broadInDimOp = subOp.rhs().getDefiningOp<mhlo::BroadcastInDimOp>();
      if (!broadInDimOp)
        return failure();

      auto maybeConstOp = getBroadcastedConstOp(broadInDimOp, featureDim);
      if (!maybeConstOp.hasValue())
        return failure();
      ConstantOp constOp = maybeConstOp.getValue();

      OpBuilder builder(subOp);
      // replace b_const with (- b_const)
      NegOp negOp = builder.create<mhlo::NegOp>(
          constOp->getLoc(), constOp.output().getType(), constOp.output());
      negOp->moveBefore(broadInDimOp);
      broadInDimOp->setOperand(0, negOp.getResult());

      // replace mhlo.sub with mhlo.add
      AddOp addOp = builder.create<mhlo::AddOp>(
          subOp->getLoc(), subOp.result().getType(), subOp.lhs(), subOp.rhs());
      subOp.result().replaceAllUsesWith(addOp.result());

    } else if (auto divOp = dyn_cast_or_null<DivOp>(convOrBiasUser)) {
      // conv_or_bias / a => conv_or_bias * (1 / a)

      // b_const should be rhs
      auto broadInDimOp = divOp.rhs().getDefiningOp<mhlo::BroadcastInDimOp>();
      if (!broadInDimOp)
        return failure();

      auto maybeConstOp = getBroadcastedConstOp(broadInDimOp, featureDim);
      if (!maybeConstOp.hasValue())
        return failure();
      ConstantOp constOp = maybeConstOp.getValue();

      OpBuilder builder(divOp);
      // replace b_const with 1 / b_const
      auto constType = constOp.output().getType().cast<RankedTensorType>();
      auto fpType = constType.getElementType().dyn_cast<FloatType>();
      if (!fpType) {
        return failure();
      }
      llvm::APFloat one(static_cast<double>(1));
      bool losesInfo; // didn't check this
      one.convert(fpType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                  &losesInfo);
      ConstantOp constOne = builder.create<mhlo::ConstantOp>(
          constOp->getLoc(), DenseFPElementsAttr::get(constType, one));
      constOne->moveBefore(broadInDimOp);
      DivOp oneDiv = builder.create<mhlo::DivOp>(
          constOp->getLoc(), constType, constOne.output(), constOp.output());
      oneDiv->moveBefore(broadInDimOp);
      broadInDimOp->setOperand(0, oneDiv.result());

      // replace mhlo.div with mhlo.mul
      MulOp mulOp = builder.create<mhlo::MulOp>(
          divOp->getLoc(), divOp.result().getType(), divOp.lhs(), divOp.rhs());
      divOp.result().replaceAllUsesWith(mulOp.result());

    } else {
      return failure();
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// PadReduceWindowToReduceWindow Pattern
//===----------------------------------------------------------------------===//

struct PadReduceWindowToReduceWindowPattern
    : public OpRewritePattern<mhlo::ReduceWindowOp> {
  using OpRewritePattern<mhlo::ReduceWindowOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ReduceWindowOp op,
                                PatternRewriter &rewriter) const override {
    if (op.operands().size() != 1 || op.init_values().size() != 1 ||
        op.getResults().size() != 1) {
      return failure();
    }
    // handle a common, special case of ReduceWindow for 1 input, 1 init_values,
    // and 1 result
    if (auto pad = dyn_cast_or_null<mhlo::PadOp>(
            op.operands().front().getDefiningOp())) {
      if (pad.padding_value() == op.init_values().front() &&
          isZeroAttribute(pad.interior_padding())) {
        // create a padding
        const auto edge_padding_low =
            pad.edge_padding_low().getValues<int64_t>();
        const auto edge_padding_high =
            pad.edge_padding_high().getValues<int64_t>();
        SmallVector<int64_t> oldPadding(edge_padding_low.size() * 2, 0);
        if (op.padding().hasValue()) {
          oldPadding = SmallVector<int64_t>(
              op.paddingAttr().getValues<int64_t>().begin(),
              op.paddingAttr().getValues<int64_t>().end());
        }
        SmallVector<int64_t> newPadding;
        for (size_t i = 0; i < edge_padding_low.size(); i++) {
          newPadding.push_back(oldPadding[i * 2] + edge_padding_low[i]);
          newPadding.push_back(oldPadding[i * 2 + 1] + edge_padding_high[i]);
        }

        auto newPaddingAttr = DenseIntElementsAttr::get(
            RankedTensorType::get(
                {static_cast<int64_t>(edge_padding_low.size()), 2},
                rewriter.getI64Type()),
            newPadding);

        auto newOp = cast<mhlo::ReduceWindowOp>(rewriter.clone(*op));
        newOp.setOperand(0, pad.operand());
        newOp.paddingAttr(newPaddingAttr);
        rewriter.replaceOp(op, newOp->getResult(0));
        return success();
      }
    }

    return failure();
  }
};

struct HloFolderPass : public HloFolderBase<HloFolderPass> {
  void runOnOperation() override {
    DimFromBroadcast dim_from_broadcast;
    DimFlagAnalysis dim_from_broadcast_analysis(&dim_from_broadcast);
    func::FuncOp funcOp = getOperation();
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
  patterns.add<PadConvToConvPattern>(patterns.getContext());
  patterns.add<ConvOrConvBiasFollowedByBroadcastOp>(patterns.getContext());
  patterns.add<PadReduceWindowToReduceWindowPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createHloFolderPass() {
  return std::make_unique<HloFolderPass>();
}