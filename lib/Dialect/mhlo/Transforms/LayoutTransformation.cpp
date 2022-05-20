//===- LayoutTransformation.cpp -------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/LayoutTransformation.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

Value createNCHW2NHWCValue(PatternRewriter &rewriter, Location loc,
                           Value input) {
  auto input_type = input.getType().cast<RankedTensorType>();
  assert(input_type.getRank() == 4);
  auto shape = input_type.getShape();
  RankedTensorType new_type = RankedTensorType::get(
      {shape[0], shape[2], shape[3], shape[1]}, input_type.getElementType());
  return rewriter.create<mhlo::TransposeOp>(
      loc, new_type, input, rewriter.getI64TensorAttr({0, 2, 3, 1}));
}

Value createNHWC2NCHWValue(PatternRewriter &rewriter, Location loc,
                           Value input) {
  auto input_type = input.getType().cast<RankedTensorType>();
  assert(input_type.getRank() == 4);
  auto shape = input_type.getShape();
  RankedTensorType new_type = RankedTensorType::get(
      {shape[0], shape[3], shape[1], shape[2]}, input_type.getElementType());
  return rewriter.create<mhlo::TransposeOp>(
      loc, new_type, input, rewriter.getI64TensorAttr({0, 3, 1, 2}));
}

RankedTensorType createNCHW2NHWCType(Type type) {
  auto type_ = type.cast<RankedTensorType>();
  assert(type_.getRank() == 4);
  auto shape = type_.getShape();
  return RankedTensorType::get({shape[0], shape[2], shape[3], shape[1]},
                               type_.getElementType());
}

DenseIntElementsAttr createNCHW2NHWCAttr(PatternRewriter &rewriter,
                                         DenseIntElementsAttr attr) {
  if (!attr) {
    return attr;
  }
  auto values = attr.getValues<int64_t>();
  assert(values.size() == 4);
  return rewriter.getI64TensorAttr(
      {values[0], values[2], values[3], values[1]});
}

DenseIntElementsAttr createNCHW2NHWCAttr2(PatternRewriter &rewriter,
                                          DenseIntElementsAttr attr) {
  if (!attr) {
    return attr;
  }
  auto values = attr.getValues<int64_t>();
  assert(values.size() == 4 * 2);
  return GetI64ElementsAttr({values[0], values[1], values[4], values[5],
                             values[6], values[7], values[2], values[3]},
                            {4, 2}, &rewriter);
}

Value createNCDHW2NDHWCValue(PatternRewriter &rewriter, Location loc,
                             Value input) {
  auto input_type = input.getType().cast<RankedTensorType>();
  assert(input_type.getRank() == 5);
  auto shape = input_type.getShape();
  RankedTensorType new_type =
      RankedTensorType::get({shape[0], shape[2], shape[3], shape[4], shape[1]},
                            input_type.getElementType());
  return rewriter.create<mhlo::TransposeOp>(
      loc, new_type, input, rewriter.getI64TensorAttr({0, 2, 3, 4, 1}));
}

Value createNDHWC2NCDHWValue(PatternRewriter &rewriter, Location loc,
                             Value input) {
  auto input_type = input.getType().cast<RankedTensorType>();
  assert(input_type.getRank() == 5);
  auto shape = input_type.getShape();
  RankedTensorType new_type =
      RankedTensorType::get({shape[0], shape[4], shape[1], shape[2], shape[3]},
                            input_type.getElementType());
  return rewriter.create<mhlo::TransposeOp>(
      loc, new_type, input, rewriter.getI64TensorAttr({0, 4, 1, 2, 3}));
}

RankedTensorType createNCDHW2NDHWCType(Type type) {
  auto type_ = type.cast<RankedTensorType>();
  assert(type_.getRank() == 5);
  auto shape = type_.getShape();
  return RankedTensorType::get(
      {shape[0], shape[2], shape[3], shape[4], shape[1]},
      type_.getElementType());
}

DenseIntElementsAttr createNCDHW2NDHWCAttr(PatternRewriter &rewriter,
                                           DenseIntElementsAttr attr) {
  if (!attr) {
    return attr;
  }
  auto values = attr.getValues<int64_t>();
  assert(values.size() == 5);
  return rewriter.getI64TensorAttr(
      {values[0], values[2], values[3], values[4], values[1]});
}

DenseIntElementsAttr createNCDHW2NDHWCAttr2(PatternRewriter &rewriter,
                                            DenseIntElementsAttr attr) {
  if (!attr) {
    return attr;
  }
  auto values = attr.getValues<int64_t>();
  assert(values.size() == 5 * 2);
  return GetI64ElementsAttr({values[0], values[1], values[4], values[5],
                             values[6], values[7], values[8], values[9],
                             values[2], values[3]},
                            {5, 2}, &rewriter);
}

struct ConvLayoutTransformationPattern : public OpRewritePattern<mhlo::ConvOp> {
  ConvLayoutTransformationPattern(MLIRContext *context,
                                  std::string targetLayout)
      : OpRewritePattern<mhlo::ConvOp>(context), targetLayout(targetLayout) {}

  LogicalResult matchAndRewrite(mhlo::ConvOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    auto dimension_numbers = op.dimension_numbers();
    auto conv_layout = getConvLayout(dimension_numbers);
    auto input_layout = std::get<0>(conv_layout);
    auto kernel_layout = std::get<1>(conv_layout);
    auto output_layout = std::get<2>(conv_layout);

    if (targetLayout == "NHWC") {
      if (input_layout == "NCHW" && kernel_layout == "NCHW" &&
          output_layout == "NCHW") {
        Value lhs_transpose =
            createNCHW2NHWCValue(rewriter, op->getLoc(), op.lhs());
        Value rhs_transpose =
            createNCHW2NHWCValue(rewriter, op->getLoc(), op.rhs());
        Type output_type = createNCHW2NHWCType(op.getResult().getType());
        auto new_dimension_numbers = mhlo::ConvDimensionNumbersAttr::get(
            rewriter.getContext(), 0, 3, {1, 2}, 3, 0, {1, 2}, 0, 3, {1, 2});
        auto newOp = rewriter.create<mhlo::ConvOp>(
            op->getLoc(), output_type, lhs_transpose, rhs_transpose,
            op.window_stridesAttr(), op.paddingAttr(), op.lhs_dilationAttr(),
            op.rhs_dilationAttr(), op.window_reversalAttr(),
            new_dimension_numbers, op.feature_group_countAttr(),
            op.batch_group_countAttr(), op.precision_configAttr());
        Value output_transpose =
            createNHWC2NCHWValue(rewriter, op->getLoc(), newOp.getResult());
        rewriter.replaceOp(op, output_transpose);
        return success();
      }
    } else if (targetLayout == "NDHWC") {
      if (input_layout == "NCDHW" && kernel_layout == "NCDHW" &&
          output_layout == "NCDHW") {
        Value lhs_transpose =
            createNCDHW2NDHWCValue(rewriter, op->getLoc(), op.lhs());
        Value rhs_transpose =
            createNCDHW2NDHWCValue(rewriter, op->getLoc(), op.rhs());
        Type output_type = createNCDHW2NDHWCType(op.getResult().getType());
        auto new_dimension_numbers = mhlo::ConvDimensionNumbersAttr::get(
            rewriter.getContext(), 0, 4, {1, 2, 3}, 4, 0, {1, 2, 3}, 0, 4,
            {1, 2, 3});
        auto newOp = rewriter.create<mhlo::ConvOp>(
            op->getLoc(), output_type, lhs_transpose, rhs_transpose,
            op.window_stridesAttr(), op.paddingAttr(), op.lhs_dilationAttr(),
            op.rhs_dilationAttr(), op.window_reversalAttr(),
            new_dimension_numbers, op.feature_group_countAttr(),
            op.batch_group_countAttr(), op.precision_configAttr());
        Value output_transpose =
            createNDHWC2NCDHWValue(rewriter, op->getLoc(), newOp.getResult());
        rewriter.replaceOp(op, output_transpose);
        return success();
      }
    }
    return failure();
  }
  std::string targetLayout;
};

struct ConvBackwardLayoutTransformationPattern
    : public OpRewritePattern<mhlo::FusionOp> {
  ConvBackwardLayoutTransformationPattern(MLIRContext *context,
                                          std::string targetLayout)
      : OpRewritePattern<mhlo::FusionOp>(context), targetLayout(targetLayout) {}

  LogicalResult matchAndRewrite(mhlo::FusionOp op,
                                PatternRewriter &rewriter) const override {
    StringAttr compute_name =
        op->getAttrOfType<StringAttr>(byre::getByreComputeName());
    if (!compute_name) {
      return failure();
    }
    if (compute_name.getValue() != "ConvBackwardDataOp" &&
        compute_name.getValue() != "ConvBackwardFilterOp") {
      return failure();
    }
    auto input_layout =
        op->getAttrOfType<StringAttr>(byre::getByrePrefix() + "input_layout")
            .getValue();
    auto kernel_layout =
        op->getAttrOfType<StringAttr>(byre::getByrePrefix() + "kernel_layout")
            .getValue();
    auto output_layout =
        op->getAttrOfType<StringAttr>(byre::getByrePrefix() + "output_layout")
            .getValue();

    if (targetLayout == "NHWC") {
      if (input_layout == "NCHW" && kernel_layout == "NCHW" &&
          output_layout == "NCHW") {
        Value lhs_transpose =
            createNCHW2NHWCValue(rewriter, op->getLoc(), op->getOperand(0));
        Value rhs_transpose =
            createNCHW2NHWCValue(rewriter, op->getLoc(), op->getOperand(1));
        Value lhs = createNHWC2NCHWValue(rewriter, op->getLoc(), lhs_transpose);
        Value rhs = createNHWC2NCHWValue(rewriter, op->getLoc(), rhs_transpose);
        Type output_type = createNCHW2NHWCType(op->getResult(0).getType());
        auto newOp = rewriter.create<mhlo::FusionOp>(
            op->getLoc(), ArrayRef<Type>{output_type},
            ArrayRef<Value>{lhs_transpose, rhs_transpose}, op->getAttrs());
        newOp->setAttr(byre::getByrePrefix() + "input_layout",
                       rewriter.getStringAttr("NHWC"));
        newOp->setAttr(byre::getByrePrefix() + "kernel_layout",
                       rewriter.getStringAttr("NHWC"));
        newOp->setAttr(byre::getByrePrefix() + "output_layout",
                       rewriter.getStringAttr("NHWC"));
        Value output_transpose =
            createNHWC2NCHWValue(rewriter, op->getLoc(), newOp->getResult(0));
        BlockAndValueMapping bvm;
        bvm.map(op->getOperand(0), lhs);
        bvm.map(op->getOperand(1), rhs);
        op.fused_computation().cloneInto(&newOp.fused_computation(), bvm);
        Block &block = newOp.fused_computation().front();
        {
          for (auto &_op : block) {
            if (llvm::isa<mhlo::ReturnOp>(&_op)) {
              OpBuilder::InsertionGuard guard(rewriter);
              rewriter.setInsertionPoint(&_op);
              Value output = createNCHW2NHWCValue(rewriter, op->getLoc(),
                                                  _op.getOperand(0));
              _op.setOperand(0, output);
            }
          }
          rhs.getDefiningOp()->moveBefore(&block.front());
          lhs.getDefiningOp()->moveBefore(&block.front());
        }

        rewriter.replaceOp(op, output_transpose);
        return success();
      }
    }
    return failure();
  }
  std::string targetLayout;
};

struct ReduceWindownLayoutTransformationPattern
    : public OpRewritePattern<mhlo::ReduceWindowOp> {
  ReduceWindownLayoutTransformationPattern(MLIRContext *context,
                                           std::string targetLayout)
      : OpRewritePattern<mhlo::ReduceWindowOp>(context),
        targetLayout(targetLayout) {}
  LogicalResult matchAndRewrite(mhlo::ReduceWindowOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    if (op.inputs().size() != 1 || op.init_values().size() != 1 ||
        op->getResults().size() != 1) {
      return failure();
    }
    auto operand = *(op.inputs().begin());
    auto layout = getPoolLayout(op);

    if (targetLayout == "NHWC" && layout == "NCHW") {
      Value operand_transpose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), operand);
      Type output_type = createNCHW2NHWCType(op->getResults()[0].getType());
      auto newOp = rewriter.create<mhlo::ReduceWindowOp>(
          op->getLoc(), ArrayRef<Type>{output_type},
          ArrayRef<Value>{operand_transpose}, op.init_values(),
          createNCHW2NHWCAttr(rewriter, op.window_dimensionsAttr()),
          createNCHW2NHWCAttr(rewriter, op.window_stridesAttr()),
          createNCHW2NHWCAttr(rewriter, op.base_dilationsAttr()),
          createNCHW2NHWCAttr(rewriter, op.window_dilationsAttr()),
          createNCHW2NHWCAttr2(rewriter, op.paddingAttr()));
      // clone body
      BlockAndValueMapping empty_map;
      op.body().cloneInto(&newOp.body(), empty_map);
      Value output_transpose =
          createNHWC2NCHWValue(rewriter, op->getLoc(), newOp->getResults()[0]);
      rewriter.replaceOp(op, output_transpose);
      return success();
    } else if (targetLayout == "NDHWC" && layout == "NCDHW") {
      Value operand_transpose =
          createNCDHW2NDHWCValue(rewriter, op->getLoc(), operand);
      Type output_type = createNCDHW2NDHWCType(op->getResults()[0].getType());
      auto newOp = rewriter.create<mhlo::ReduceWindowOp>(
          op->getLoc(), ArrayRef<Type>{output_type},
          ArrayRef<Value>{operand_transpose}, op.init_values(),
          createNCDHW2NDHWCAttr(rewriter, op.window_dimensionsAttr()),
          createNCDHW2NDHWCAttr(rewriter, op.window_stridesAttr()),
          createNCDHW2NDHWCAttr(rewriter, op.base_dilationsAttr()),
          createNCDHW2NDHWCAttr(rewriter, op.window_dilationsAttr()),
          createNCDHW2NDHWCAttr2(rewriter, op.paddingAttr()));
      // clone body
      BlockAndValueMapping empty_map;
      op.body().cloneInto(&newOp.body(), empty_map);
      Value output_transpose = createNDHWC2NCDHWValue(rewriter, op->getLoc(),
                                                      newOp->getResults()[0]);
      rewriter.replaceOp(op, output_transpose);
      return success();
    }
    return failure();
  }
  std::string targetLayout;
};

struct SelectAndScatterLayoutTransformationPattern
    : public OpRewritePattern<mhlo::SelectAndScatterOp> {
  SelectAndScatterLayoutTransformationPattern(MLIRContext *context,
                                              std::string targetLayout)
      : OpRewritePattern<mhlo::SelectAndScatterOp>(context),
        targetLayout(targetLayout) {}
  LogicalResult matchAndRewrite(mhlo::SelectAndScatterOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    auto layout = getPoolGradLayout(op);

    if (targetLayout == "NHWC" && layout == "NCHW") {
      Value operand_transpose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.operand());
      Value source_transpose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.source());
      Type output_type = createNCHW2NHWCType(op.getResult().getType());
      auto newOp = rewriter.create<mhlo::SelectAndScatterOp>(
          op->getLoc(), output_type, operand_transpose, source_transpose,
          op.init_value(),
          createNCHW2NHWCAttr(rewriter, op.window_dimensionsAttr()),
          createNCHW2NHWCAttr(rewriter, op.window_stridesAttr()),
          createNCHW2NHWCAttr2(rewriter, op.paddingAttr()));
      // clone body
      BlockAndValueMapping empty_map;
      op.select().cloneInto(&newOp.select(), empty_map);
      op.scatter().cloneInto(&newOp.scatter(), empty_map);
      Value output_transpose =
          createNHWC2NCHWValue(rewriter, op->getLoc(), newOp.getResult());
      rewriter.replaceOp(op, output_transpose);
      return success();
    } else if (targetLayout == "NDHWC" && layout == "NCDHW") {
      Value operand_transpose =
          createNCDHW2NDHWCValue(rewriter, op->getLoc(), op.operand());
      Value source_transpose =
          createNCDHW2NDHWCValue(rewriter, op->getLoc(), op.source());
      Type output_type = createNCDHW2NDHWCType(op.getResult().getType());
      auto newOp = rewriter.create<mhlo::SelectAndScatterOp>(
          op->getLoc(), output_type, operand_transpose, source_transpose,
          op.init_value(),
          createNCDHW2NDHWCAttr(rewriter, op.window_dimensionsAttr()),
          createNCDHW2NDHWCAttr(rewriter, op.window_stridesAttr()),
          createNCDHW2NDHWCAttr2(rewriter, op.paddingAttr()));
      // clone body
      BlockAndValueMapping empty_map;
      op.select().cloneInto(&newOp.select(), empty_map);
      op.scatter().cloneInto(&newOp.scatter(), empty_map);
      Value output_transpose =
          createNDHWC2NCDHWValue(rewriter, op->getLoc(), newOp.getResult());
      rewriter.replaceOp(op, output_transpose);
      return success();
    }
    return failure();
  }
  std::string targetLayout;
};

struct BatchNormTrainingLayoutTransformationPattern
    : public OpRewritePattern<mhlo::BatchNormTrainingOp> {
  BatchNormTrainingLayoutTransformationPattern(MLIRContext *context,
                                               std::string targetLayout)
      : OpRewritePattern<mhlo::BatchNormTrainingOp>(context),
        targetLayout(targetLayout) {}

  LogicalResult matchAndRewrite(mhlo::BatchNormTrainingOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    auto input_type = op.operand().getType().cast<RankedTensorType>();
    if (targetLayout == "NHWC" && input_type.getRank() == 4 &&
        op.feature_index() == 1) {
      Value input_transpose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.operand());
      Type output_type = createNCHW2NHWCType(op.output().getType());
      mhlo::BatchNormTrainingOp op_transpose =
          rewriter.create<mhlo::BatchNormTrainingOp>(
              op->getLoc(),
              ArrayRef<Type>{output_type, op.batch_mean().getType(),
                             op.batch_var().getType()},
              input_transpose, op.scale(), op.offset(), op.epsilonAttr(),
              rewriter.getI64IntegerAttr(3));
      Value output_transpose =
          createNHWC2NCHWValue(rewriter, op->getLoc(), op_transpose.output());

      rewriter.replaceOp(op, {output_transpose, op_transpose.batch_mean(),
                              op_transpose.batch_var()});
      return success();
    } else {
      return failure();
    }
  }
  std::string targetLayout;
};

struct BatchNormGradLayoutTransformationPattern
    : public OpRewritePattern<mhlo::BatchNormGradOp> {
  BatchNormGradLayoutTransformationPattern(MLIRContext *context,
                                           std::string targetLayout)
      : OpRewritePattern<mhlo::BatchNormGradOp>(context),
        targetLayout(targetLayout) {}
  LogicalResult matchAndRewrite(mhlo::BatchNormGradOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    auto input_type = op.operand().getType().cast<RankedTensorType>();
    if (targetLayout == "NHWC" && input_type.getRank() == 4 &&
        op.feature_index() == 1) {
      Value operand_transpose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.operand());
      Value grad_output_transpose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.grad_output());
      Type grad_operand_type = createNCHW2NHWCType(op.grad_operand().getType());
      mhlo::BatchNormGradOp op_transpose =
          rewriter.create<mhlo::BatchNormGradOp>(
              op->getLoc(),
              ArrayRef<Type>{grad_operand_type, op.grad_scale().getType(),
                             op.grad_offset().getType()},
              operand_transpose, op.scale(), op.mean(), op.variance(),
              grad_output_transpose, op.epsilonAttr(),
              rewriter.getI64IntegerAttr(3));
      Value output_transpose = createNHWC2NCHWValue(
          rewriter, op->getLoc(), op_transpose.grad_operand());
      rewriter.replaceOp(op, {output_transpose, op_transpose.grad_scale(),
                              op_transpose.grad_offset()});
      return success();
    }
    return failure();
  }
  std::string targetLayout;
};

struct LayoutTransformationPass
    : LayoutTransformationBase<LayoutTransformationPass> {
  LayoutTransformationPass(std::string target_layout)
      : LayoutTransformationBase() {
    this->targetLayout = target_layout;
  }

  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    if (this->targetLayout != "NHWC" && this->targetLayout != "NDHWC") {
      funcOp.emitError(
          "LayoutTransformationPass doesn't support target layout: ")
          << this->targetLayout;
      return signalPassFailure();
    }

    RewritePatternSet patterns(funcOp.getContext());
    populateLayoutTransformationPattern(patterns, this->targetLayout);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError("LayoutTransformationPass applyPatternsAndFoldGreedily "
                       "does not converge");
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateLayoutTransformationPattern(RewritePatternSet &patterns,
                                               std::string targetLayout) {
  patterns.add<ConvLayoutTransformationPattern,
               ConvBackwardLayoutTransformationPattern,
               ReduceWindownLayoutTransformationPattern,
               SelectAndScatterLayoutTransformationPattern,
               BatchNormTrainingLayoutTransformationPattern,
               BatchNormGradLayoutTransformationPattern>(patterns.getContext(),
                                                         targetLayout);
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLayoutTransformationPass(std::string target_layout) {
  return std::make_unique<LayoutTransformationPass>(target_layout);
}