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
    return constOp.getValue().isSplat();
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
    if (auto denseIntE = constOp.getValue().dyn_cast<DenseIntElementsAttr>()) {
      return isSplatValue(denseIntE, splat_val);
    }
  }
  return false;
}

bool mlir::isSplatMhloConstantValue(Operation *op, double splat_val) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstantOp>(op)) {
    // only handle DenseFPElementsAttr for now
    // TODO extend it
    if (auto denseFPE = constOp.getValue().dyn_cast<DenseFPElementsAttr>()) {
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

static byteir::NamedLayout
parsePoolLayout(size_t rank, const SmallVector<int64_t> &window_dimensions,
                const SmallVector<int64_t> &strides,
                const SmallVector<int64_t> &padding) {
  byteir::NamedLayout layout = byteir::NamedLayout::UNKNOWN;
  if (window_dimensions[0] == 1 && window_dimensions[rank - 1] == 1 &&
      strides[0] == 1 && strides[rank - 1] == 1 && padding[0] == 0 &&
      padding[1] == 0 && padding[2 * rank - 2] == 0 &&
      padding[2 * rank - 1] == 0) {
    if (rank == 4) {
      layout = byteir::NamedLayout::NHWC;
    } else if (rank == 5) {
      layout = byteir::NamedLayout::NDHWC;
    }
  } else if (window_dimensions[0] == 1 && window_dimensions[1] == 1 &&
             strides[0] == 1 && strides[1] == 1 && padding[0] == 0 &&
             padding[1] == 0 && padding[2] == 0 && padding[3] == 0) {
    if (rank == 4) {
      layout = byteir::NamedLayout::NCHW;
    } else if (rank == 5) {
      layout = byteir::NamedLayout::NCDHW;
    }
  }
  return layout;
}

byteir::NamedLayout mlir::getPoolLayout(mlir::mhlo::ReduceWindowOp op) {
  auto base_dilations = op.getBaseDilationsAttr();
  if (base_dilations && !isSplatValue(base_dilations, 1)) {
    assert(false && "expected base_dilations to be dense<1>");
  }
  auto window_dilations = op.getWindowDilationsAttr();
  if (window_dilations && !isSplatValue(window_dilations, 1)) {
    assert(false && "expected window_dilations to be dense<1>");
  }

  SmallVector<int64_t> window_dimensions = SmallVector<int64_t>(
      op.getWindowDimensions().getValues<int64_t>().begin(),
      op.getWindowDimensions().getValues<int64_t>().end());
  size_t rank = window_dimensions.size();
  SmallVector<int64_t> strides(rank, 1);
  if (auto strides_ = op.getWindowStridesAttr()) {
    strides = SmallVector<int64_t>(strides_.getValues<int64_t>().begin(),
                                   strides_.getValues<int64_t>().end());
  }
  SmallVector<int64_t> padding(rank * 2, 0);
  if (auto padding_ = op.getPaddingAttr()) {
    padding = SmallVector<int64_t>(padding_.getValues<int64_t>().begin(),
                                   padding_.getValues<int64_t>().end());
  }

  if (rank != 4 && rank != 5) {
    assert(false && "expected dimension number to be 4 or 5");
  }
  return parsePoolLayout(rank, window_dimensions, strides, padding);
}

byteir::NamedLayout mlir::getPoolGradLayout(mlir::mhlo::SelectAndScatterOp op) {
  SmallVector<int64_t> window_dimensions;
  if (auto window_dimensions_ = op.getWindowDimensionsAttr()) {
    window_dimensions =
        SmallVector<int64_t>(window_dimensions_.getValues<int64_t>().begin(),
                             window_dimensions_.getValues<int64_t>().end());
  }
  size_t rank = window_dimensions.size();
  SmallVector<int64_t> strides(rank, 1);
  if (auto window_strides = op.getWindowStridesAttr()) {
    strides = SmallVector<int64_t>(window_strides.getValues<int64_t>().begin(),
                                   window_strides.getValues<int64_t>().end());
  }
  SmallVector<int64_t> padding(rank * 2, 0);
  if (auto padding_ = op.getPaddingAttr()) {
    padding = SmallVector<int64_t>(padding_.getValues<int64_t>().begin(),
                                   padding_.getValues<int64_t>().end());
  }

  assert(rank == 4 || rank == 5);
  return parsePoolLayout(rank, window_dimensions, strides, padding);
}

std::tuple<byteir::NamedLayout, byteir::NamedLayout, byteir::NamedLayout>
mlir::getConvLayout(mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers) {
  byteir::NamedLayout input_layout = byteir::NamedLayout::UNKNOWN;
  auto input_batch_dimension = dimension_numbers.getInputBatchDimension();
  auto input_feature_dimension = dimension_numbers.getInputFeatureDimension();
  auto input_spatial_dimensions = dimension_numbers.getInputSpatialDimensions();
  if (input_spatial_dimensions.size() == 1) {
    if (input_batch_dimension == 0 && input_feature_dimension == 1) {
      input_layout = byteir::NamedLayout::NCW;
    } else {
      input_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else if (input_spatial_dimensions.size() == 2) {
    if (input_batch_dimension == 0 && input_feature_dimension == 1) {
      input_layout = byteir::NamedLayout::NCHW;
    } else if (input_batch_dimension == 0 && input_feature_dimension == 3) {
      input_layout = byteir::NamedLayout::NHWC;
    } else {
      input_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else if (input_spatial_dimensions.size() == 3) {
    if (input_batch_dimension == 0 && input_feature_dimension == 1) {
      input_layout = byteir::NamedLayout::NCDHW;
    } else if (input_batch_dimension == 0 && input_feature_dimension == 4) {
      input_layout = byteir::NamedLayout::NDHWC;
    } else {
      input_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else {
    input_layout = byteir::NamedLayout::UNKNOWN;
  }

  byteir::NamedLayout output_layout;
  auto output_batch_dimension = dimension_numbers.getOutputBatchDimension();
  auto output_feature_dimension = dimension_numbers.getOutputFeatureDimension();
  auto output_spatial_dimensions =
      dimension_numbers.getOutputSpatialDimensions();
  if (output_spatial_dimensions.size() == 1) {
    if (output_batch_dimension == 0 && output_feature_dimension == 1) {
      output_layout = byteir::NamedLayout::NCW;
    } else {
      output_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else if (output_spatial_dimensions.size() == 2) {
    if (output_batch_dimension == 0 && output_feature_dimension == 1) {
      output_layout = byteir::NamedLayout::NCHW;
    } else if (output_batch_dimension == 0 && output_feature_dimension == 3) {
      output_layout = byteir::NamedLayout::NHWC;
    } else {
      output_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else if (output_spatial_dimensions.size() == 3) {
    if (output_batch_dimension == 0 && output_feature_dimension == 1) {
      output_layout = byteir::NamedLayout::NCDHW;
    } else if (output_batch_dimension == 0 && output_feature_dimension == 4) {
      output_layout = byteir::NamedLayout::NDHWC;
    } else {
      output_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else {
    output_layout = byteir::NamedLayout::UNKNOWN;
  }

  byteir::NamedLayout kernel_layout;
  auto kernel_input_feature_dimension =
      dimension_numbers.getKernelInputFeatureDimension();
  auto kernel_output_feature_dimension =
      dimension_numbers.getKernelOutputFeatureDimension();
  auto kernel_spatial_dimensions =
      dimension_numbers.getKernelSpatialDimensions();
  if (kernel_spatial_dimensions.size() == 1) {
    if (kernel_input_feature_dimension == 1 &&
        kernel_output_feature_dimension == 0) {
      kernel_layout = byteir::NamedLayout::NCW;
    } else {
      kernel_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else if (kernel_spatial_dimensions.size() == 2) {
    if (kernel_input_feature_dimension == 1 &&
        kernel_output_feature_dimension == 0) {
      kernel_layout = byteir::NamedLayout::NCHW;
    } else if (kernel_input_feature_dimension == 3 &&
               kernel_output_feature_dimension == 0) {
      kernel_layout = byteir::NamedLayout::NHWC;
    } else if (kernel_input_feature_dimension == 2 &&
               kernel_output_feature_dimension == 3) {
      kernel_layout = byteir::NamedLayout::HWCN;
    } else {
      kernel_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else if (kernel_spatial_dimensions.size() == 3) {
    if (kernel_input_feature_dimension == 1 &&
        kernel_output_feature_dimension == 0) {
      kernel_layout = byteir::NamedLayout::NCDHW;
    } else if (kernel_input_feature_dimension == 4 &&
               kernel_output_feature_dimension == 0) {
      kernel_layout = byteir::NamedLayout::NDHWC;
    } else if (kernel_input_feature_dimension == 3 &&
               kernel_output_feature_dimension == 4) {
      kernel_layout = byteir::NamedLayout::DHWCN;
    } else {
      kernel_layout = byteir::NamedLayout::UNKNOWN;
    }
  } else {
    kernel_layout = byteir::NamedLayout::UNKNOWN;
  }

  return std::make_tuple(input_layout, kernel_layout, output_layout);
}

template <typename T>
void mlir::handleConvAttribute(NamedAttrList &attrs, T conv_op,
                               OpBuilder &rewriter) {
  auto dimension_numbers = conv_op.getDimensionNumbers();
  auto conv_layout = mlir::getConvLayout(dimension_numbers);

  auto input_layout = std::get<0>(conv_layout);
  auto kernel_layout = std::get<1>(conv_layout);
  auto output_layout = std::get<2>(conv_layout);
  assert(input_layout != byteir::NamedLayout::UNKNOWN &&
         kernel_layout != byteir::NamedLayout::UNKNOWN &&
         output_layout != byteir::NamedLayout::UNKNOWN);
  assert(input_layout == kernel_layout && input_layout == output_layout);

  attrs.append("input_layout",
               rewriter.getStringAttr(byteir::stringifyEnum(input_layout)));
  attrs.append("output_layout",
               rewriter.getStringAttr(byteir::stringifyEnum(output_layout)));
  attrs.append("kernel_layout",
               rewriter.getStringAttr(byteir::stringifyEnum(kernel_layout)));

  if (conv_op.getWindowStrides().hasValue()) {
    attrs.append("window_strides", conv_op.getWindowStridesAttr());
  }
  if (conv_op.getPadding().hasValue()) {
    attrs.append("padding", conv_op.getPaddingAttr());
  }
  if (conv_op.getLhsDilation().hasValue()) {
    attrs.append("lhs_dilation", conv_op.getLhsDilationAttr());
  }
  if (conv_op.getRhsDilation().hasValue()) {
    attrs.append("rhs_dilation", conv_op.getRhsDilationAttr());
  }
  attrs.append("feature_group_count", conv_op.getFeatureGroupCountAttr());
  attrs.append("batch_group_count", conv_op.getBatchGroupCountAttr());
  if (conv_op.getWindowReversal().hasValue()) {
    attrs.append("window_reversal", conv_op.getWindowReversalAttr());
  }
}

template void mlir::handleConvAttribute<mhlo::ConvolutionOp>(
    NamedAttrList &, mhlo::ConvolutionOp, OpBuilder &);
template void mlir::handleConvAttribute<lmhlo::ConvolutionOp>(
    NamedAttrList &, lmhlo::ConvolutionOp, OpBuilder &);

#undef UNKNOWN_LAYOUT
