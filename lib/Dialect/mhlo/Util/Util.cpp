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
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_attrs.h"
#include "mlir/IR/Operation.h"

using namespace llvm;

bool mlir::IsSplatMhloConstant(Operation *op) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstOp>(op)) {
    return constOp.value().isSplat();
  }
  return false;
}

bool mlir::IsSplatMhloConstantValue(Operation *op, int64_t splat_val) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstOp>(op)) {
    // only handle DenseFPElementsAttr for now
    // TODO extend it
    if (auto denseIntE = constOp.value().dyn_cast<DenseIntElementsAttr>()) {
      return isSplatValue(denseIntE, splat_val);
    }
  }
  return false;
}

bool mlir::IsSplatMhloConstantValue(Operation *op, double splat_val) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstOp>(op)) {
    // only handle DenseFPElementsAttr for now
    // TODO extend it
    if (auto denseFPE = constOp.value().dyn_cast<DenseFPElementsAttr>()) {
      return isSplatValue(denseFPE, splat_val);
    }
  }
  return false;
}

bool mlir::IsSplatMhloConstantValue(Value val, int64_t splat_val) {
  return IsSplatMhloConstantValue(val.getDefiningOp(), splat_val);
}

bool mlir::IsSplatMhloConstantValue(Value val, double splat_val) {
  return IsSplatMhloConstantValue(val.getDefiningOp(), splat_val);
}

// TODO: make this a template later for max/min
bool mlir::IsBlockSingleAdd(Block *block) {
  if (block == nullptr)
    return false;

  auto ret_op = block->getTerminator();
  if (!isa<mlir::mhlo::ReturnOp>(ret_op))
    return false;

  auto mhlo_ret = cast<mlir::mhlo::ReturnOp>(ret_op);
  if (mhlo_ret.getNumOperands() != 1)
    return false;

  auto compute_op = mhlo_ret.getOperand(0).getDefiningOp();
  if (auto add_op = dyn_cast_or_null<mhlo::AddOp>(compute_op)) {
    return (compute_op->getOperand(0) == block->getArgument(0) &&
            compute_op->getOperand(1) == block->getArgument(1)) ||
           (compute_op->getOperand(0) == block->getArgument(1) &&
            compute_op->getOperand(1) == block->getArgument(0));
  }

  return false;
}

template <typename T>
void mlir::HandleConvAttribute(NamedAttrList &attrs, T conv_op,
                               OpBuilder &rewriter) {
  auto dimension_numbers = conv_op.dimension_numbersAttr();

  StringAttr input_layout;
  auto input_batch_dimension = dimension_numbers.getInputBatchDimension();
  auto input_feature_dimension = dimension_numbers.getInputFeatureDimension();
  assert(dimension_numbers.getInputSpatialDimensions().size() == 2);
  if (input_batch_dimension == 0 && input_feature_dimension == 1) {
    input_layout = rewriter.getStringAttr("NCHW");
  } else if (input_batch_dimension == 0 && input_feature_dimension == 3) {
    input_layout = rewriter.getStringAttr("NHWC");
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
    kernel_layout = rewriter.getStringAttr("NCHW");
  } else if (kernel_input_feature_dimension == 3 &&
             kernel_output_feature_dimension == 0) {
    kernel_layout = rewriter.getStringAttr("NHWC");
  } else if (kernel_input_feature_dimension == 2 &&
             kernel_output_feature_dimension == 3) {
    kernel_layout = rewriter.getStringAttr("HWCN");
  } else {
    assert(false && "Unsupported convolution kernel layout.");
  }

  attrs.append("input_layout", input_layout);
  attrs.append("output_layout", output_layout);
  attrs.append("kernel_layout", kernel_layout);
  if (conv_op.window_strides()) {
    attrs.append("window_strides", conv_op.window_stridesAttr());
  }
  if (conv_op.padding()) {
    attrs.append("padding", conv_op.paddingAttr());
  }
  if (conv_op.lhs_dilation()) {
    attrs.append("lhs_dilation", conv_op.lhs_dilationAttr());
  }
  if (conv_op.rhs_dilation()) {
    attrs.append("rhs_dilation", conv_op.rhs_dilationAttr());
  }
  attrs.append("feature_group_count", conv_op.feature_group_countAttr());
  attrs.append("batch_group_count", conv_op.batch_group_countAttr());
  if (conv_op.window_reversal()) {
    attrs.append("window_reversal", conv_op.window_reversalAttr());
  }
}

template void mlir::HandleConvAttribute<mlir::mhlo::ConvOp>(NamedAttrList &,
                                                            mlir::mhlo::ConvOp,
                                                            OpBuilder &);
template void mlir::HandleConvAttribute<mlir::lmhlo::ConvOp>(
    NamedAttrList &, mlir::lmhlo::ConvOp, OpBuilder &);
