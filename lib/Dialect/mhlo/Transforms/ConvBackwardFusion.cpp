//===- ConvBackwardFusion.cpp ---------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/ConvBackwardFusion.h"
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
#include <assert.h>

using namespace mlir;
using namespace llvm;

namespace {

static void
GetConvDimensionNumbers(mhlo::ConvDimensionNumbersAttr dimension_numbers,
                        SmallVector<int64_t> &input_dims,
                        SmallVector<int64_t> &kernel_dims,
                        SmallVector<int64_t> &output_dims) {
  input_dims.push_back(dimension_numbers.getInputBatchDimension());
  input_dims.push_back(dimension_numbers.getInputFeatureDimension());
  for (auto i : dimension_numbers.getInputSpatialDimensions()) {
    input_dims.push_back(i);
  }
  kernel_dims.push_back(dimension_numbers.getKernelOutputFeatureDimension());
  kernel_dims.push_back(dimension_numbers.getKernelInputFeatureDimension());
  for (auto i : dimension_numbers.getKernelSpatialDimensions()) {
    kernel_dims.push_back(i);
  }
  output_dims.push_back(dimension_numbers.getOutputBatchDimension());
  output_dims.push_back(dimension_numbers.getOutputFeatureDimension());
  for (auto i : dimension_numbers.getOutputSpatialDimensions()) {
    output_dims.push_back(i);
  }
}

struct FuseConvBackwardDataPattern
    : public OpRewritePattern<mhlo::ConvolutionOp> {
  using OpRewritePattern<mhlo::ConvolutionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    auto dimension_numbers = op.dimension_numbersAttr();
    SmallVector<int64_t> output_grad_dims, filter_dims, input_grad_dims;
    GetConvDimensionNumbers(dimension_numbers, output_grad_dims, filter_dims,
                            input_grad_dims);
    if (output_grad_dims != ArrayRef<int64_t>{0, 1, 2, 3} /*bf01*/ ||
        filter_dims != ArrayRef<int64_t>{2, 3, 0, 1} /*01io*/ ||
        input_grad_dims != ArrayRef<int64_t>{0, 1, 2, 3} /*bf01*/) {
      return failure();
    }

    MhloFusionPattern pattern;
    SmallVector<Value> inputs, outputs;
    if (auto reverseOp = op.rhs().getDefiningOp<mhlo::ReverseOp>()) {
      SmallVector<int64_t> dimensions;
      getValuesFromDenseIntElementsAttr(reverseOp.dimensions(), dimensions);
      assert((dimensions == ArrayRef<int64_t>{0, 1}));
      if (auto transposeOp =
              reverseOp.operand().getDefiningOp<mhlo::TransposeOp>()) {
        SmallVector<int64_t> permutation;
        getValuesFromDenseIntElementsAttr(transposeOp.permutation(),
                                          permutation);
        assert((permutation == ArrayRef<int64_t>{2, 3, 1, 0}));
        inputs.push_back(op.lhs());
        inputs.push_back(transposeOp.operand());
        pattern.push_back(transposeOp);
        pattern.push_back(reverseOp);
      }
    } else if (auto transposeOp = op.rhs().getDefiningOp<mhlo::TransposeOp>()) {
      // TODO: check kH = 1 and kW = 1
      SmallVector<int64_t> permutation;
      getValuesFromDenseIntElementsAttr(transposeOp.permutation(), permutation);
      assert((permutation == ArrayRef<int64_t>{2, 3, 1, 0}));
      inputs.push_back(op.lhs());
      inputs.push_back(transposeOp.operand());
      pattern.push_back(transposeOp);
    } else {
      return failure();
    }
    pattern.push_back(op);
    outputs.push_back(op.getResult());

    mhlo::FusionOp fusionOp =
        createMhloFusionFromPattern(rewriter, inputs, outputs, pattern);

    NamedAttrList attrs;
    if (op.window_stridesAttr()) {
      SmallVector<int64_t> window_strides;
      getValuesFromDenseIntElementsAttr(op.window_stridesAttr(),
                                        window_strides);
      assert((window_strides == ArrayRef<int64_t>{1, 1}));
    }
    // TODO: handle more window_strides pattern
    int64_t stridesH, stridesW;
    if (op.lhs_dilationAttr()) {
      SmallVector<int64_t> lhs_dilation;
      getValuesFromDenseIntElementsAttr(op.lhs_dilationAttr(), lhs_dilation);
      if (lhs_dilation == ArrayRef<int64_t>{1, 1}) {
        stridesH = 1;
        stridesW = 1;
      } else if (lhs_dilation == ArrayRef<int64_t>{2, 2}) {
        stridesH = 2;
        stridesW = 2;
      } else {
        assert((false && "invalid window_strides"));
      }
    } else {
      stridesH = 1;
      stridesW = 1;
    }
    if (op.rhs_dilationAttr()) {
      SmallVector<int64_t> rhs_dilation;
      getValuesFromDenseIntElementsAttr(op.rhs_dilationAttr(), rhs_dilation);
      assert((rhs_dilation == ArrayRef<int64_t>{1, 1}));
    }

    auto input_shape = outputs[0].getType().cast<RankedTensorType>().getShape();
    auto kernel_shape = inputs[1].getType().cast<RankedTensorType>().getShape();
    auto output_shape = inputs[0].getType().cast<RankedTensorType>().getShape();
    int64_t paddingH =
        (output_shape[2] - 1) * stridesH + kernel_shape[2] - input_shape[2];
    paddingH = (paddingH + 1) / 2;
    int64_t paddingW =
        (output_shape[3] - 1) * stridesW + kernel_shape[3] - input_shape[3];
    paddingW = (paddingW + 1) / 2;

    attrs.append(byre::getByreComputeName(),
                 rewriter.getStringAttr("ConvBackwardDataOp"));
    byre::appendByreComputeAttr(attrs, "input_layout",
                                rewriter.getStringAttr("NCHW"));
    byre::appendByreComputeAttr(attrs, "kernel_layout",
                                rewriter.getStringAttr("NCHW"));
    byre::appendByreComputeAttr(attrs, "output_layout",
                                rewriter.getStringAttr("NCHW"));
    byre::appendByreComputeAttr(
        attrs, "window_strides",
        rewriter.getI64TensorAttr({stridesH, stridesW}));
    byre::appendByreComputeAttr(
        attrs, "padding",
        rewriter.getI64TensorAttr({paddingH, paddingH, paddingW, paddingW}));
    byre::appendByreComputeAttr(attrs, "feature_group_count",
                                rewriter.getI64IntegerAttr(1));
    byre::appendByreComputeAttr(attrs, "batch_group_count",
                                rewriter.getI64IntegerAttr(1));

    fusionOp->setAttrs(attrs.getDictionary(getContext()));

    return success();
  }
};

struct FuseConvBackwardFilterPattern
    : public OpRewritePattern<mhlo::TransposeOp> {
  using OpRewritePattern<mhlo::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (transposeOp->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    SmallVector<int64_t> permutation;
    getValuesFromDenseIntElementsAttr(transposeOp.permutation(), permutation);
    if (permutation != ArrayRef<int64_t>{3, 2, 0, 1}) {
      return failure();
    }

    auto op = transposeOp.operand().getDefiningOp<mhlo::ConvolutionOp>();
    if (!op) {
      return failure();
    }
    auto dimension_numbers = op.dimension_numbersAttr();
    SmallVector<int64_t> input_dims, output_grad_dims, filter_grad_dims;
    GetConvDimensionNumbers(dimension_numbers, input_dims, output_grad_dims,
                            filter_grad_dims);
    if (input_dims != ArrayRef<int64_t>{1, 0, 2, 3} /*fb01*/ ||
        output_grad_dims != ArrayRef<int64_t>{1, 0, 2, 3} /*io01*/ ||
        filter_grad_dims != ArrayRef<int64_t>{2, 3, 0, 1} /*01bf*/) {
      return failure();
    }

    SmallVector<Value> inputs{op.lhs(), op.rhs()};
    SmallVector<Value> outputs{transposeOp.getResult()};
    MhloFusionPattern pattern{op, transposeOp};

    mhlo::FusionOp fusionOp =
        createMhloFusionFromPattern(rewriter, inputs, outputs, pattern);

    NamedAttrList attrs;
    if (op.window_stridesAttr()) {
      SmallVector<int64_t> window_strides;
      getValuesFromDenseIntElementsAttr(op.window_stridesAttr(),
                                        window_strides);
      assert((window_strides == ArrayRef<int64_t>{1, 1}));
    }
    // TODO: handle more window_strides pattern
    int64_t stridesH, stridesW;
    if (op.rhs_dilationAttr()) {
      SmallVector<int64_t> rhs_dilation;
      getValuesFromDenseIntElementsAttr(op.rhs_dilationAttr(), rhs_dilation);
      if (rhs_dilation == ArrayRef<int64_t>{1, 1}) {
        stridesH = 1;
        stridesW = 1;
      } else if (rhs_dilation == ArrayRef<int64_t>{2, 2}) {
        stridesH = 2;
        stridesW = 2;
      } else {
        assert((false && "invalid window_strides"));
      }
    } else {
      stridesH = 1;
      stridesW = 1;
    }
    if (op.lhs_dilationAttr()) {
      SmallVector<int64_t> lhs_dilation;
      getValuesFromDenseIntElementsAttr(op.lhs_dilationAttr(), lhs_dilation);
      assert((lhs_dilation == ArrayRef<int64_t>{1, 1}));
    }

    auto input_shape = inputs[0].getType().cast<RankedTensorType>().getShape();
    auto kernel_shape =
        outputs[0].getType().cast<RankedTensorType>().getShape();
    auto output_shape = inputs[1].getType().cast<RankedTensorType>().getShape();
    int64_t paddingH =
        (output_shape[2] - 1) * stridesH + kernel_shape[2] - input_shape[2];
    paddingH = (paddingH + 1) / 2;
    int64_t paddingW =
        (output_shape[3] - 1) * stridesW + kernel_shape[3] - input_shape[3];
    paddingW = (paddingW + 1) / 2;

    attrs.append(byre::getByreComputeName(),
                 rewriter.getStringAttr("ConvBackwardFilterOp"));
    byre::appendByreComputeAttr(attrs, "input_layout",
                                rewriter.getStringAttr("NCHW"));
    byre::appendByreComputeAttr(attrs, "kernel_layout",
                                rewriter.getStringAttr("NCHW"));
    byre::appendByreComputeAttr(attrs, "output_layout",
                                rewriter.getStringAttr("NCHW"));
    byre::appendByreComputeAttr(
        attrs, "window_strides",
        rewriter.getI64TensorAttr({stridesH, stridesW}));
    byre::appendByreComputeAttr(
        attrs, "padding",
        rewriter.getI64TensorAttr({paddingH, paddingH, paddingW, paddingW}));
    byre::appendByreComputeAttr(attrs, "feature_group_count",
                                rewriter.getI64IntegerAttr(1));
    byre::appendByreComputeAttr(attrs, "batch_group_count",
                                rewriter.getI64IntegerAttr(1));

    fusionOp->setAttrs(attrs.getDictionary(getContext()));

    return success();
  }
};

struct ConvBackwardFusionPass
    : public ConvBackwardFusionBase<ConvBackwardFusionPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateFuseConvBackwardPatterns(patterns);
    LogicalResult status =
        applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    if (failed(status)) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateFuseConvBackwardPatterns(RewritePatternSet &patterns) {
  patterns.add(
      std::make_unique<FuseConvBackwardDataPattern>(patterns.getContext()));
  patterns.add(
      std::make_unique<FuseConvBackwardFilterPattern>(patterns.getContext()));
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvBackwardFusionPass() {
  return std::make_unique<ConvBackwardFusionPass>();
}