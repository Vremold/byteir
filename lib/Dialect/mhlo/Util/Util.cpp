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
using namespace mlir;
using namespace mlir::mhlo;

bool mlir::isMhlo(Operation *op) {
  Dialect *dialect = op->getDialect();
  return dialect && isa<MhloDialect>(dialect);
}

bool mlir::isSplatMhloConstant(Operation *op) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstOp>(op)) {
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
  return isa<mhlo::ConstOp>(op) || isa<mhlo::IotaOp>(op);
}

bool mlir::isSplatMhloConstantValue(Value val) {
  return isSplatMhloConstant(val.getDefiningOp());
}

bool mlir::isSplatMhloConstantValue(Operation *op, int64_t splat_val) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstOp>(op)) {
    // only handle DenseFPElementsAttr for now
    // TODO extend it
    if (auto denseIntE = constOp.value().dyn_cast<DenseIntElementsAttr>()) {
      return isSplatValue(denseIntE, splat_val);
    }
  }
  return false;
}

bool mlir::isSplatMhloConstantValue(Operation *op, double splat_val) {
  if (auto constOp = dyn_cast_or_null<mhlo::ConstOp>(op)) {
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

// TODO: make this a template later for max/min
bool mlir::isBlockSingleAdd(Block *block) {
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

#define UNKNOWN_STR "UNKNOWN_LAYOUT"

std::string mlir::getPoolLayout(mlir::mhlo::ReduceWindowOp op) {
  auto base_dilations = op.base_dilationsAttr();
  if (base_dilations && !isSplatValue(base_dilations, 1)) {
    assert(false && "expected base_dilations to be dense<1>");
  }
  auto window_dilations = op.window_dilationsAttr();
  if (window_dilations && !isSplatValue(window_dilations, 1)) {
    assert(false && "expected window_dilations to be dense<1>");
  }

  auto kernel = op.window_dimensions().getValues<int64_t>();
  SmallVector<int64_t> strides;
  if (auto strides_ = op.window_stridesAttr()) {
    strides = SmallVector<int64_t>(strides_.getValues<int64_t>().begin(),
                                   strides_.getValues<int64_t>().end());
  }
  SmallVector<int64_t> padding;
  if (auto padding_ = op.paddingAttr()) {
    padding = SmallVector<int64_t>(padding_.getValues<int64_t>().begin(),
                                   padding_.getValues<int64_t>().end());
  }

  int64_t rank = kernel.size();
  if (rank != 4 && rank != 5) {
    assert(false && "expected dimension number to be 4 or 5");
  }

  std::string layout = UNKNOWN_STR;
  if (kernel[0] == 1 && kernel[rank - 1] == 1) {
    if (rank == 4) {
      layout = "NHWC";
    }
    if (rank == 5) {
      layout = "NDHWC";
    }
    if (strides.size() != 0 && (strides[0] != 1 || strides[rank - 1] != 1)) {
      layout = UNKNOWN_STR;
    }
    if (layout != UNKNOWN_STR && padding.size() != 0 &&
        (padding[0] != 0 || padding[1] != 0 || padding[rank * 2 - 2] != 0 ||
         padding[rank * 2 - 1] != 0)) {
      layout = UNKNOWN_STR;
    }
  }
  if (layout == UNKNOWN_STR && kernel[0] == 1 && kernel[1] == 1) {
    if (rank == 4) {
      layout = "NCHW";
    }
    if (rank == 5) {
      layout = "NCDHW";
    }
    if (strides.size() != 0 && (strides[0] != 1 || strides[1] != 1)) {
      layout = UNKNOWN_STR;
    }
    if (layout != UNKNOWN_STR && padding.size() != 0 &&
        (padding[0] != 0 || padding[1] != 0 || padding[2] != 0 ||
         padding[3] != 0)) {
      layout = UNKNOWN_STR;
    }
  }
  return layout;
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
      input_layout = UNKNOWN_STR;
    }
  } else if (input_spatial_dimensions.size() == 3) {
    if (input_batch_dimension == 0 && input_feature_dimension == 1) {
      input_layout = "NCDHW";
    } else if (input_batch_dimension == 0 && input_feature_dimension == 4) {
      input_layout = "NDHWC";
    } else {
      input_layout = UNKNOWN_STR;
    }
  } else {
    input_layout = UNKNOWN_STR;
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
      output_layout = UNKNOWN_STR;
    }
  } else if (output_spatial_dimensions.size() == 3) {
    if (output_batch_dimension == 0 && output_feature_dimension == 1) {
      output_layout = "NCDHW";
    } else if (output_batch_dimension == 0 && output_feature_dimension == 4) {
      output_layout = "NDHWC";
    } else {
      output_layout = UNKNOWN_STR;
    }
  } else {
    output_layout = UNKNOWN_STR;
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
      kernel_layout = UNKNOWN_STR;
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
      kernel_layout = UNKNOWN_STR;
    }
  } else {
    kernel_layout = UNKNOWN_STR;
  }

  return std::make_tuple(input_layout, kernel_layout, output_layout);
}

template <typename T>
void mlir::handleConvAttribute(NamedAttrList &attrs, T conv_op,
                               OpBuilder &rewriter) {
  auto dimension_numbers = conv_op.dimension_numbersAttr();
  auto conv_layout = mlir::getConvLayout(dimension_numbers);

  auto input_layout = std::get<0>(conv_layout);
  auto kernel_layout = std::get<1>(conv_layout);
  auto output_layout = std::get<2>(conv_layout);
  assert(input_layout != UNKNOWN_STR && kernel_layout != UNKNOWN_STR &&
         output_layout != UNKNOWN_STR);
  assert(input_layout == kernel_layout && input_layout == output_layout);

  attrs.append("input_layout", rewriter.getStringAttr(input_layout));
  attrs.append("output_layout", rewriter.getStringAttr(output_layout));
  attrs.append("kernel_layout", rewriter.getStringAttr(kernel_layout));

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

template void mlir::handleConvAttribute<mlir::mhlo::ConvOp>(NamedAttrList &,
                                                            mlir::mhlo::ConvOp,
                                                            OpBuilder &);
template void mlir::handleConvAttribute<mlir::lmhlo::ConvOp>(
    NamedAttrList &, mlir::lmhlo::ConvOp, OpBuilder &);

#undef UNKNOWN_STR
