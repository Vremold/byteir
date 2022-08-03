//===- Util.cpp -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Operation.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

bool mlir::isMhlo(Operation *op) {
  Dialect *dialect = op->getDialect();
  return dialect && isa<MhloDialect>(dialect);
}

bool mlir::isSplatMhloConstant(Operation *op) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstantOp>(op)) {
    return constOp.value().isSplat();
  }
  return false;
}

bool mlir::isSplatMhloConstantLike(Operation *op) {
  return isSplatMhloConstant(op) || isa_and_nonnull<mhlo::IotaOp>(op);
}

bool mlir::isMhloConstantLike(Operation *op) {
  if (!op)
    return false;
  return isa<mhlo::ConstantOp>(op) || isa<mhlo::IotaOp>(op);
}

bool mlir::isSplatMhloConstantValue(Value val) {
  return isSplatMhloConstant(val.getDefiningOp());
}

bool mlir::isSplatMhloConstantValue(Operation *op, int64_t splat_val) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstantOp>(op)) {
    // only handle DenseFPElementsAttr for now
    // TODO extend it
    if (auto denseIntE = constOp.value().dyn_cast<DenseIntElementsAttr>()) {
      return isSplatValue(denseIntE, splat_val);
    }
  }
  return false;
}

bool mlir::isSplatMhloConstantValue(Operation *op, double splat_val) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstantOp>(op)) {
    // only handle DenseFPElementsAttr for now
    // TODO extend it
    if (auto denseFPE = constOp.value().dyn_cast<DenseFPElementsAttr>()) {
      return isSplatValue(denseFPE, splat_val);
    }
  }
  return false;
}

bool mlir::isSplatMhloConstantValue(Value val, int64_t splat_val) {
  return isSplatMhloConstantValue(val.getDefiningOp(), splat_val);
}

bool mlir::isSplatMhloConstantValue(Value val, double splat_val) {
  return isSplatMhloConstantValue(val.getDefiningOp(), splat_val);
}

template <typename Op> bool mlir::isBlockSingleOp(Block *block) {
  if (block == nullptr)
    return false;

  auto ret_op = block->getTerminator();
  if (!isa<mlir::mhlo::ReturnOp>(ret_op))
    return false;

  auto mhlo_ret = cast<mlir::mhlo::ReturnOp>(ret_op);
  if (mhlo_ret.getNumOperands() != 1)
    return false;

  auto compute_op = mhlo_ret.getOperand(0).getDefiningOp();
  if (auto add_op = dyn_cast_or_null<Op>(compute_op)) {
    return (compute_op->getOperand(0) == block->getArgument(0) &&
            compute_op->getOperand(1) == block->getArgument(1)) ||
           (compute_op->getOperand(0) == block->getArgument(1) &&
            compute_op->getOperand(1) == block->getArgument(0));
  }

  return false;
}

template bool mlir::isBlockSingleOp<mhlo::AddOp>(Block *);
template bool mlir::isBlockSingleOp<mhlo::MaxOp>(Block *);
template bool mlir::isBlockSingleOp<mhlo::MinOp>(Block *);

#define UNKNOWN_LAYOUT "UNKNOWN"

static std::string
parsePoolLayout(size_t rank, const SmallVector<int64_t> &window_dimensions,
                const SmallVector<int64_t> &strides,
                const SmallVector<int64_t> &padding) {
  std::string layout = UNKNOWN_LAYOUT;
  if (window_dimensions[0] == 1 && window_dimensions[rank - 1] == 1 &&
      strides[0] == 1 && strides[rank - 1] == 1 && padding[0] == 0 &&
      padding[1] == 0 && padding[2 * rank - 2] == 0 &&
      padding[2 * rank - 1] == 0) {
    if (rank == 4) {
      layout = "NHWC";
    } else if (rank == 5) {
      layout = "NDHWC";
    }
  } else if (window_dimensions[0] == 1 && window_dimensions[1] == 1 &&
             strides[0] == 1 && strides[1] == 1 && padding[0] == 0 &&
             padding[1] == 0 && padding[2] == 0 && padding[3] == 0) {
    if (rank == 4) {
      layout = "NCHW";
    } else if (rank == 5) {
      layout = "NCDHW";
    }
  }
  return layout;
}

std::string mlir::getPoolLayout(mlir::mhlo::ReduceWindowOp op) {
  auto base_dilations = op.base_dilationsAttr();
  if (base_dilations && !isSplatValue(base_dilations, 1)) {
    assert(false && "expected base_dilations to be dense<1>");
  }
  auto window_dilations = op.window_dilationsAttr();
  if (window_dilations && !isSplatValue(window_dilations, 1)) {
    assert(false && "expected window_dilations to be dense<1>");
  }

  SmallVector<int64_t> window_dimensions =
      SmallVector<int64_t>(op.window_dimensions().getValues<int64_t>().begin(),
                           op.window_dimensions().getValues<int64_t>().end());
  size_t rank = window_dimensions.size();
  SmallVector<int64_t> strides(rank, 1);
  if (auto strides_ = op.window_stridesAttr()) {
    strides = SmallVector<int64_t>(strides_.getValues<int64_t>().begin(),
                                   strides_.getValues<int64_t>().end());
  }
  SmallVector<int64_t> padding(rank * 2, 0);
  if (auto padding_ = op.paddingAttr()) {
    padding = SmallVector<int64_t>(padding_.getValues<int64_t>().begin(),
                                   padding_.getValues<int64_t>().end());
  }

  if (rank != 4 && rank != 5) {
    assert(false && "expected dimension number to be 4 or 5");
  }
  return parsePoolLayout(rank, window_dimensions, strides, padding);
}

std::string mlir::getPoolGradLayout(mlir::mhlo::SelectAndScatterOp op) {
  std::string layout = UNKNOWN_LAYOUT;
  SmallVector<int64_t> window_dimensions;
  if (auto window_dimensions_ = op.window_dimensionsAttr()) {
    window_dimensions =
        SmallVector<int64_t>(window_dimensions_.getValues<int64_t>().begin(),
                             window_dimensions_.getValues<int64_t>().end());
  }
  size_t rank = window_dimensions.size();
  SmallVector<int64_t> strides(rank, 1);
  if (auto window_strides = op.window_stridesAttr()) {
    strides = SmallVector<int64_t>(window_strides.getValues<int64_t>().begin(),
                                   window_strides.getValues<int64_t>().end());
  }
  SmallVector<int64_t> padding(rank * 2, 0);
  if (auto padding_ = op.paddingAttr()) {
    padding = SmallVector<int64_t>(padding_.getValues<int64_t>().begin(),
                                   padding_.getValues<int64_t>().end());
  }

  assert(rank == 4 || rank == 5);
  return parsePoolLayout(rank, window_dimensions, strides, padding);
}

std::tuple<std::string, std::string, std::string>
mlir::getConvLayout(mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers) {
  std::string input_layout;
  auto input_batch_dimension = dimension_numbers.getInputBatchDimension();
  auto input_feature_dimension = dimension_numbers.getInputFeatureDimension();
  auto input_spatial_dimensions = dimension_numbers.getInputSpatialDimensions();
  if (input_spatial_dimensions.size() == 2) {
    if (input_batch_dimension == 0 && input_feature_dimension == 1) {
      input_layout = "NCHW";
    } else if (input_batch_dimension == 0 && input_feature_dimension == 3) {
      input_layout = "NHWC";
    } else {
      input_layout = UNKNOWN_LAYOUT;
    }
  } else if (input_spatial_dimensions.size() == 3) {
    if (input_batch_dimension == 0 && input_feature_dimension == 1) {
      input_layout = "NCDHW";
    } else if (input_batch_dimension == 0 && input_feature_dimension == 4) {
      input_layout = "NDHWC";
    } else {
      input_layout = UNKNOWN_LAYOUT;
    }
  } else {
    input_layout = UNKNOWN_LAYOUT;
  }

  std::string output_layout;
  auto output_batch_dimension = dimension_numbers.getOutputBatchDimension();
  auto output_feature_dimension = dimension_numbers.getOutputFeatureDimension();
  auto output_spatial_dimensions =
      dimension_numbers.getOutputSpatialDimensions();
  if (output_spatial_dimensions.size() == 2) {
    if (output_batch_dimension == 0 && output_feature_dimension == 1) {
      output_layout = "NCHW";
    } else if (output_batch_dimension == 0 && output_feature_dimension == 3) {
      output_layout = "NHWC";
    } else {
      output_layout = UNKNOWN_LAYOUT;
    }
  } else if (output_spatial_dimensions.size() == 3) {
    if (output_batch_dimension == 0 && output_feature_dimension == 1) {
      output_layout = "NCDHW";
    } else if (output_batch_dimension == 0 && output_feature_dimension == 4) {
      output_layout = "NDHWC";
    } else {
      output_layout = UNKNOWN_LAYOUT;
    }
  } else {
    output_layout = UNKNOWN_LAYOUT;
  }

  std::string kernel_layout;
  auto kernel_input_feature_dimension =
      dimension_numbers.getKernelInputFeatureDimension();
  auto kernel_output_feature_dimension =
      dimension_numbers.getKernelOutputFeatureDimension();
  auto kernel_spatial_dimensions =
      dimension_numbers.getKernelSpatialDimensions();
  if (kernel_spatial_dimensions.size() == 2) {
    if (kernel_input_feature_dimension == 1 &&
        kernel_output_feature_dimension == 0) {
      kernel_layout = "NCHW";
    } else if (kernel_input_feature_dimension == 3 &&
               kernel_output_feature_dimension == 0) {
      kernel_layout = "NHWC";
    } else if (kernel_input_feature_dimension == 2 &&
               kernel_output_feature_dimension == 3) {
      kernel_layout = "HWCN";
    } else {
      kernel_layout = UNKNOWN_LAYOUT;
    }
  } else if (kernel_spatial_dimensions.size() == 3) {
    if (kernel_input_feature_dimension == 1 &&
        kernel_output_feature_dimension == 0) {
      kernel_layout = "NCDHW";
    } else if (kernel_input_feature_dimension == 4 &&
               kernel_output_feature_dimension == 0) {
      kernel_layout = "NDHWC";
    } else if (kernel_input_feature_dimension == 3 &&
               kernel_output_feature_dimension == 4) {
      kernel_layout = "DHWCN";
    } else {
      kernel_layout = UNKNOWN_LAYOUT;
    }
  } else {
    kernel_layout = UNKNOWN_LAYOUT;
  }

  return std::make_tuple(input_layout, kernel_layout, output_layout);
}

template <typename T>
void mlir::handleConvAttribute(NamedAttrList &attrs, T conv_op,
                               OpBuilder &rewriter) {
  auto dimension_numbers =
      conv_op->template getAttrOfType<mhlo::ConvDimensionNumbersAttr>(
          "dimension_numbers");
  auto conv_layout = mlir::getConvLayout(dimension_numbers);

  auto input_layout = std::get<0>(conv_layout);
  auto kernel_layout = std::get<1>(conv_layout);
  auto output_layout = std::get<2>(conv_layout);
  assert(input_layout != UNKNOWN_LAYOUT && kernel_layout != UNKNOWN_LAYOUT &&
         output_layout != UNKNOWN_LAYOUT);
  assert(input_layout == kernel_layout && input_layout == output_layout);

  attrs.append("input_layout", rewriter.getStringAttr(input_layout));
  attrs.append("output_layout", rewriter.getStringAttr(output_layout));
  attrs.append("kernel_layout", rewriter.getStringAttr(kernel_layout));

  if (conv_op->hasAttr("window_strides")) {
    attrs.append("window_strides", conv_op->getAttr("window_strides"));
  }
  if (conv_op->hasAttr("padding")) {
    attrs.append("padding", conv_op->getAttr("padding"));
  }
  if (conv_op->hasAttr("lhs_dilation")) {
    attrs.append("lhs_dilation", conv_op->getAttr("lhs_dilation"));
  }
  if (conv_op->hasAttr("rhs_dilation")) {
    attrs.append("rhs_dilation", conv_op->getAttr("rhs_dilation"));
  }
  attrs.append("feature_group_count", conv_op->getAttr("feature_group_count"));
  attrs.append("batch_group_count", conv_op->getAttr("batch_group_count"));
  if (conv_op->hasAttr("window_reversal")) {
    attrs.append("window_reversal", conv_op->getAttr("window_reversal"));
  }
}

template void mlir::handleConvAttribute<mhlo::ConvolutionOp>(
    NamedAttrList &, mhlo::ConvolutionOp, OpBuilder &);
template void mlir::handleConvAttribute<lmhlo::ConvolutionOp>(
    NamedAttrList &, lmhlo::ConvolutionOp, OpBuilder &);

#undef UNKNOWN_LAYOUT
