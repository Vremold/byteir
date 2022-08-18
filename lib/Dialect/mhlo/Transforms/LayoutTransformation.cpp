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
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  auto inputType = input.getType().cast<RankedTensorType>();
  assert(inputType.getRank() == 4);
  auto shape = inputType.getShape();
  RankedTensorType newType = RankedTensorType::get(
      {shape[0], shape[2], shape[3], shape[1]}, inputType.getElementType());
  return rewriter.create<mhlo::TransposeOp>(
      loc, newType, input, rewriter.getI64TensorAttr({0, 2, 3, 1}));
}

Value createNHWC2NCHWValue(PatternRewriter &rewriter, Location loc,
                           Value input) {
  auto inputType = input.getType().cast<RankedTensorType>();
  assert(inputType.getRank() == 4);
  auto shape = inputType.getShape();
  RankedTensorType newType = RankedTensorType::get(
      {shape[0], shape[3], shape[1], shape[2]}, inputType.getElementType());
  return rewriter.create<mhlo::TransposeOp>(
      loc, newType, input, rewriter.getI64TensorAttr({0, 3, 1, 2}));
}

Value createHWCN2NHWCValue(PatternRewriter &rewriter, Location loc,
                           Value input) {
  auto inputType = input.getType().cast<RankedTensorType>();
  assert(inputType.getRank() == 4);
  auto shape = inputType.getShape();
  RankedTensorType newType = RankedTensorType::get(
      {shape[3], shape[0], shape[1], shape[2]}, inputType.getElementType());
  return rewriter.create<mhlo::TransposeOp>(
      loc, newType, input, rewriter.getI64TensorAttr({3, 0, 1, 2}));
}

RankedTensorType createNCHW2NHWCType(Type type) {
  auto rankedTy = type.cast<RankedTensorType>();
  assert(rankedTy.getRank() == 4);
  auto shape = rankedTy.getShape();
  return RankedTensorType::get({shape[0], shape[2], shape[3], shape[1]},
                               rankedTy.getElementType());
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
  return getI64ElementsAttr({values[0], values[1], values[4], values[5],
                             values[6], values[7], values[2], values[3]},
                            {4, 2}, &rewriter);
}

Value createNCDHW2NDHWCValue(PatternRewriter &rewriter, Location loc,
                             Value input) {
  auto inputType = input.getType().cast<RankedTensorType>();
  assert(inputType.getRank() == 5);
  auto shape = inputType.getShape();
  RankedTensorType newType =
      RankedTensorType::get({shape[0], shape[2], shape[3], shape[4], shape[1]},
                            inputType.getElementType());
  return rewriter.create<mhlo::TransposeOp>(
      loc, newType, input, rewriter.getI64TensorAttr({0, 2, 3, 4, 1}));
}

Value createNDHWC2NCDHWValue(PatternRewriter &rewriter, Location loc,
                             Value input) {
  auto inputType = input.getType().cast<RankedTensorType>();
  assert(inputType.getRank() == 5);
  auto shape = inputType.getShape();
  RankedTensorType newType =
      RankedTensorType::get({shape[0], shape[4], shape[1], shape[2], shape[3]},
                            inputType.getElementType());
  return rewriter.create<mhlo::TransposeOp>(
      loc, newType, input, rewriter.getI64TensorAttr({0, 4, 1, 2, 3}));
}

RankedTensorType createNCDHW2NDHWCType(Type type) {
  auto rankedTy = type.cast<RankedTensorType>();
  assert(rankedTy.getRank() == 5);
  auto shape = rankedTy.getShape();
  return RankedTensorType::get(
      {shape[0], shape[2], shape[3], shape[4], shape[1]},
      rankedTy.getElementType());
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
  return getI64ElementsAttr({values[0], values[1], values[4], values[5],
                             values[6], values[7], values[8], values[9],
                             values[2], values[3]},
                            {5, 2}, &rewriter);
}

struct ConvLayoutTransformationPattern
    : public OpRewritePattern<mhlo::ConvolutionOp> {
  ConvLayoutTransformationPattern(MLIRContext *context,
                                  std::string targetLayout)
      : OpRewritePattern<mhlo::ConvolutionOp>(context),
        targetLayout(targetLayout) {}

  LogicalResult matchAndRewrite(mhlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    auto dimensionNumbers = op.dimension_numbers();
    auto convLayout = getConvLayout(dimensionNumbers);
    auto inputLayout = std::get<0>(convLayout);
    auto kernelLayout = std::get<1>(convLayout);
    auto outputLayout = std::get<2>(convLayout);

    if (targetLayout == "NHWC") {
      if (inputLayout == "NCHW" && kernelLayout == "NCHW" &&
          outputLayout == "NCHW") {
        Value lhsTranspose =
            createNCHW2NHWCValue(rewriter, op->getLoc(), op.lhs());
        Value rhsTranspose =
            createNCHW2NHWCValue(rewriter, op->getLoc(), op.rhs());
        Type outputType = createNCHW2NHWCType(op.getResult().getType());
        auto newDimensionNumbers = mhlo::ConvDimensionNumbersAttr::get(
            rewriter.getContext(), 0, 3, {1, 2}, 3, 0, {1, 2}, 0, 3, {1, 2});
        auto newOp = rewriter.create<mhlo::ConvolutionOp>(
            op->getLoc(), outputType, lhsTranspose, rhsTranspose,
            op.window_stridesAttr(), op.paddingAttr(), op.lhs_dilationAttr(),
            op.rhs_dilationAttr(), op.window_reversalAttr(),
            newDimensionNumbers, op.feature_group_countAttr(),
            op.batch_group_countAttr(), op.precision_configAttr());
        Value outputTranspose =
            createNHWC2NCHWValue(rewriter, op->getLoc(), newOp.getResult());
        rewriter.replaceOp(op, outputTranspose);
        return success();
      } else if (inputLayout == "NHWC" && kernelLayout == "HWCN" &&
                 outputLayout == "NHWC") {
        Value rhsTranspose =
            createHWCN2NHWCValue(rewriter, op->getLoc(), op.rhs());
        auto newDimensionNumbers = mhlo::ConvDimensionNumbersAttr::get(
            rewriter.getContext(), 0, 3, {1, 2}, 3, 0, {1, 2}, 0, 3, {1, 2});
        mhlo::ConvolutionOp newOp = rewriter.create<mhlo::ConvolutionOp>(
            op->getLoc(), op.getType(), op.lhs(), rhsTranspose,
            op.window_stridesAttr(), op.paddingAttr(), op.lhs_dilationAttr(),
            op.rhs_dilationAttr(), op.window_reversalAttr(),
            newDimensionNumbers, op.feature_group_countAttr(),
            op.batch_group_countAttr(), op.precision_configAttr());
        rewriter.replaceOp(op, newOp.getResult());
        return success();
      }
    } else if (targetLayout == "NDHWC") {
      if (inputLayout == "NCDHW" && kernelLayout == "NCDHW" &&
          outputLayout == "NCDHW") {
        Value lhsTranspose =
            createNCDHW2NDHWCValue(rewriter, op->getLoc(), op.lhs());
        Value rhsTranspose =
            createNCDHW2NDHWCValue(rewriter, op->getLoc(), op.rhs());
        Type outputType = createNCDHW2NDHWCType(op.getResult().getType());
        auto newDimensionNumbers = mhlo::ConvDimensionNumbersAttr::get(
            rewriter.getContext(), 0, 4, {1, 2, 3}, 4, 0, {1, 2, 3}, 0, 4,
            {1, 2, 3});
        auto newOp = rewriter.create<mhlo::ConvolutionOp>(
            op->getLoc(), outputType, lhsTranspose, rhsTranspose,
            op.window_stridesAttr(), op.paddingAttr(), op.lhs_dilationAttr(),
            op.rhs_dilationAttr(), op.window_reversalAttr(),
            newDimensionNumbers, op.feature_group_countAttr(),
            op.batch_group_countAttr(), op.precision_configAttr());
        Value outputTranspose =
            createNDHWC2NCDHWValue(rewriter, op->getLoc(), newOp.getResult());
        rewriter.replaceOp(op, outputTranspose);
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
    StringAttr computeName =
        op->getAttrOfType<StringAttr>(byre::getByreComputeName());
    if (!computeName) {
      return failure();
    }
    if (computeName.getValue() != "ConvBackwardDataOp" &&
        computeName.getValue() != "ConvBackwardFilterOp") {
      return failure();
    }
    auto inputLayout =
        op->getAttrOfType<StringAttr>(byre::getByrePrefix() + "input_layout")
            .getValue();
    auto kernelLayout =
        op->getAttrOfType<StringAttr>(byre::getByrePrefix() + "kernel_layout")
            .getValue();
    auto outputLayout =
        op->getAttrOfType<StringAttr>(byre::getByrePrefix() + "output_layout")
            .getValue();

    if (targetLayout == "NHWC") {
      if (inputLayout == "NCHW" && kernelLayout == "NCHW" &&
          outputLayout == "NCHW") {
        Value lhsTranspose =
            createNCHW2NHWCValue(rewriter, op->getLoc(), op->getOperand(0));
        Value rhsTranspose =
            createNCHW2NHWCValue(rewriter, op->getLoc(), op->getOperand(1));
        Value lhs = createNHWC2NCHWValue(rewriter, op->getLoc(), lhsTranspose);
        Value rhs = createNHWC2NCHWValue(rewriter, op->getLoc(), rhsTranspose);
        Type outputType = createNCHW2NHWCType(op->getResult(0).getType());
        auto newOp = rewriter.create<mhlo::FusionOp>(
            op->getLoc(), ArrayRef<Type>{outputType},
            ArrayRef<Value>{lhsTranspose, rhsTranspose}, op->getAttrs());
        newOp->setAttr(byre::getByrePrefix() + "input_layout",
                       rewriter.getStringAttr("NHWC"));
        newOp->setAttr(byre::getByrePrefix() + "kernel_layout",
                       rewriter.getStringAttr("NHWC"));
        newOp->setAttr(byre::getByrePrefix() + "output_layout",
                       rewriter.getStringAttr("NHWC"));
        Value outputTranspose =
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

        rewriter.replaceOp(op, outputTranspose);
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
    if (op.operands().size() != 1 || op.init_values().size() != 1 ||
        op->getResults().size() != 1) {
      return failure();
    }
    auto operand = *(op.operands().begin());
    auto layout = getPoolLayout(op);

    if (targetLayout == "NHWC" && layout == "NCHW") {
      Value operandTranspose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), operand);
      Type outputType = createNCHW2NHWCType(op->getResults()[0].getType());
      auto newOp = rewriter.create<mhlo::ReduceWindowOp>(
          op->getLoc(), ArrayRef<Type>{outputType},
          ArrayRef<Value>{operandTranspose}, op.init_values(),
          createNCHW2NHWCAttr(rewriter, op.window_dimensionsAttr()),
          createNCHW2NHWCAttr(rewriter, op.window_stridesAttr()),
          createNCHW2NHWCAttr(rewriter, op.base_dilationsAttr()),
          createNCHW2NHWCAttr(rewriter, op.window_dilationsAttr()),
          createNCHW2NHWCAttr2(rewriter, op.paddingAttr()));
      // clone body
      BlockAndValueMapping emptyBvm;
      op.body().cloneInto(&newOp.body(), emptyBvm);
      Value outputTranspose =
          createNHWC2NCHWValue(rewriter, op->getLoc(), newOp->getResults()[0]);
      rewriter.replaceOp(op, outputTranspose);
      return success();
    } else if (targetLayout == "NDHWC" && layout == "NCDHW") {
      Value operandTranspose =
          createNCDHW2NDHWCValue(rewriter, op->getLoc(), operand);
      Type outputType = createNCDHW2NDHWCType(op->getResults()[0].getType());
      auto newOp = rewriter.create<mhlo::ReduceWindowOp>(
          op->getLoc(), ArrayRef<Type>{outputType},
          ArrayRef<Value>{operandTranspose}, op.init_values(),
          createNCDHW2NDHWCAttr(rewriter, op.window_dimensionsAttr()),
          createNCDHW2NDHWCAttr(rewriter, op.window_stridesAttr()),
          createNCDHW2NDHWCAttr(rewriter, op.base_dilationsAttr()),
          createNCDHW2NDHWCAttr(rewriter, op.window_dilationsAttr()),
          createNCDHW2NDHWCAttr2(rewriter, op.paddingAttr()));
      // clone body
      BlockAndValueMapping emptyBvm;
      op.body().cloneInto(&newOp.body(), emptyBvm);
      Value outputTranspose = createNDHWC2NCDHWValue(rewriter, op->getLoc(),
                                                     newOp->getResults()[0]);
      rewriter.replaceOp(op, outputTranspose);
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
      Value operandTranspose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.operand());
      Value sourceTranspose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.source());
      Type outputType = createNCHW2NHWCType(op.getResult().getType());
      auto newOp = rewriter.create<mhlo::SelectAndScatterOp>(
          op->getLoc(), outputType, operandTranspose, sourceTranspose,
          op.init_value(),
          createNCHW2NHWCAttr(rewriter, op.window_dimensionsAttr()),
          createNCHW2NHWCAttr(rewriter, op.window_stridesAttr()),
          createNCHW2NHWCAttr2(rewriter, op.paddingAttr()));
      // clone body
      BlockAndValueMapping emptyBvm;
      op.select().cloneInto(&newOp.select(), emptyBvm);
      op.scatter().cloneInto(&newOp.scatter(), emptyBvm);
      Value outputTranspose =
          createNHWC2NCHWValue(rewriter, op->getLoc(), newOp.getResult());
      rewriter.replaceOp(op, outputTranspose);
      return success();
    } else if (targetLayout == "NDHWC" && layout == "NCDHW") {
      Value operandTranspose =
          createNCDHW2NDHWCValue(rewriter, op->getLoc(), op.operand());
      Value sourceTranspose =
          createNCDHW2NDHWCValue(rewriter, op->getLoc(), op.source());
      Type outputType = createNCDHW2NDHWCType(op.getResult().getType());
      auto newOp = rewriter.create<mhlo::SelectAndScatterOp>(
          op->getLoc(), outputType, operandTranspose, sourceTranspose,
          op.init_value(),
          createNCDHW2NDHWCAttr(rewriter, op.window_dimensionsAttr()),
          createNCDHW2NDHWCAttr(rewriter, op.window_stridesAttr()),
          createNCDHW2NDHWCAttr2(rewriter, op.paddingAttr()));
      // clone body
      BlockAndValueMapping emptyBvm;
      op.select().cloneInto(&newOp.select(), emptyBvm);
      op.scatter().cloneInto(&newOp.scatter(), emptyBvm);
      Value outputTranspose =
          createNDHWC2NCDHWValue(rewriter, op->getLoc(), newOp.getResult());
      rewriter.replaceOp(op, outputTranspose);
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
    auto inputType = op.operand().getType().cast<RankedTensorType>();
    if (targetLayout == "NHWC" && inputType.getRank() == 4 &&
        op.feature_index() == 1) {
      Value inputTranspose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.operand());
      Type outputType = createNCHW2NHWCType(op.output().getType());
      mhlo::BatchNormTrainingOp opTranspose =
          rewriter.create<mhlo::BatchNormTrainingOp>(
              op->getLoc(),
              ArrayRef<Type>{outputType, op.batch_mean().getType(),
                             op.batch_var().getType()},
              inputTranspose, op.scale(), op.offset(), op.epsilonAttr(),
              rewriter.getI64IntegerAttr(3));
      Value outputTranspose =
          createNHWC2NCHWValue(rewriter, op->getLoc(), opTranspose.output());

      rewriter.replaceOp(op, {outputTranspose, opTranspose.batch_mean(),
                              opTranspose.batch_var()});
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
    auto inputType = op.operand().getType().cast<RankedTensorType>();
    if (targetLayout == "NHWC" && inputType.getRank() == 4 &&
        op.feature_index() == 1) {
      Value operandTranspose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.operand());
      Value gradOutputTranspose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.grad_output());
      Type gradOperandType = createNCHW2NHWCType(op.grad_operand().getType());
      mhlo::BatchNormGradOp opTranspose =
          rewriter.create<mhlo::BatchNormGradOp>(
              op->getLoc(),
              ArrayRef<Type>{gradOperandType, op.grad_scale().getType(),
                             op.grad_offset().getType()},
              operandTranspose, op.scale(), op.mean(), op.variance(),
              gradOutputTranspose, op.epsilonAttr(),
              rewriter.getI64IntegerAttr(3));
      Value outputTranspose = createNHWC2NCHWValue(rewriter, op->getLoc(),
                                                   opTranspose.grad_operand());
      rewriter.replaceOp(op, {outputTranspose, opTranspose.grad_scale(),
                              opTranspose.grad_offset()});
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
    func::FuncOp funcOp = getOperation();
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

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLayoutTransformationPass(std::string target_layout) {
  return std::make_unique<LayoutTransformationPass>(target_layout);
}