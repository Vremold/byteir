//===- Common.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/ToByre/Common.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Parser.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::lmhlo;

mlir::LogicalResult mlir::ConvertToByrePattern<mlir::CallOp>::matchAndRewrite(
    mlir::CallOp op, typename mlir::CallOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto funcOp = GetFuncOp(op);
  if (funcOp == nullptr) {
    return failure();
  }

  StringAttr nameAttr =
      funcOp->getAttrOfType<StringAttr>(byre::getByreComputeName());

  if (nameAttr == nullptr) {
    return failure();
  }

  mlir::byre::ComputeOp computeOp =
      rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(op, nameAttr.getValue(), adaptor.getOperands());

  SmallVector<NamedAttribute> attrs;
  for (auto iter = funcOp->getAttrs().begin(); iter != funcOp->getAttrs().end(); iter++) {
    if (byre::isByreComputeAttr(*iter)) {
      attrs.emplace_back(byre::removeByrePrefix(*iter));
    }
  }

  AddAttrs(computeOp.getOperation(), attrs);

  return success();
}

mlir::LogicalResult
mlir::ConvertToByrePattern<mlir::lmhlo::DotOp>::matchAndRewrite(
    mlir::lmhlo::DotOp op, typename mlir::lmhlo::DotOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto dot_dimension_numbers = adaptor.dot_dimension_numbers();
  assert(dot_dimension_numbers.getLhsContractingDimensions().size() == 1);
  assert(dot_dimension_numbers.getRhsContractingDimensions().size() == 1);
  if (dot_dimension_numbers.getLhsBatchingDimensions().size() == 0) {
    // convert to MatmulOp
    auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
        op, "MatmulOp", adaptor.getOperands());

    // append attribute 'lhs_contracting_dimension' and
    // 'rhs_contracting_dimension'
    int64_t lhs_contracting_dimension =
        dot_dimension_numbers.getLhsContractingDimensions()[0];
    int64_t rhs_contracting_dimension =
        dot_dimension_numbers.getRhsContractingDimensions()[0];
    compute_op->setAttr("lhs_contracting_dimension",
                        rewriter.getI64IntegerAttr(lhs_contracting_dimension));
    compute_op->setAttr("rhs_contracting_dimension",
                        rewriter.getI64IntegerAttr(rhs_contracting_dimension));
  } else {
    // convert to BatchMatmulOp
    SmallVector<int64_t> batching_dimensions;
    for (int64_t i = 0, e = op.output().getType().cast<ShapedType>().getRank();
         i < e - 2; i++) {
      batching_dimensions.push_back(i);
    }
    if (!dot_dimension_numbers.getLhsBatchingDimensions().equals(
            batching_dimensions) ||
        !dot_dimension_numbers.getRhsBatchingDimensions().equals(
            batching_dimensions)) {
      return op->emitOpError()
             << "can not handle unregular batching_dimensions";
    }

    auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
        op, "BatchMatmulOp", adaptor.getOperands());

    // append attributes of batching and contracting dimensions
    int64_t lhs_contracting_dimension =
        dot_dimension_numbers.getLhsContractingDimensions()[0];
    int64_t rhs_contracting_dimension =
        dot_dimension_numbers.getRhsContractingDimensions()[0];
    auto lhs_batching_dimensions =
        dot_dimension_numbers.getLhsBatchingDimensions();
    auto rhs_batching_dimensions =
        dot_dimension_numbers.getRhsBatchingDimensions();
    compute_op->setAttr("lhs_contracting_dimension",
                        rewriter.getI64IntegerAttr(lhs_contracting_dimension));
    compute_op->setAttr("rhs_contracting_dimension",
                        rewriter.getI64IntegerAttr(rhs_contracting_dimension));
    compute_op->setAttr("lhs_batching_dimensions",
                        rewriter.getI64ArrayAttr(lhs_batching_dimensions));
    compute_op->setAttr("rhs_batching_dimensions",
                        rewriter.getI64ArrayAttr(rhs_batching_dimensions));
  }
  return success();
}

mlir::LogicalResult
mlir::ConvertToByrePattern<lmhlo::CustomCallOp>::matchAndRewrite(
    lmhlo::CustomCallOp op, typename lmhlo::CustomCallOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  mlir::DictionaryAttr dict_attr;
  auto backend_config = op.backend_config();
  if (!backend_config.empty()) {
    auto attrs = mlir::parseAttribute(backend_config, op->getContext());
    if (!attrs || !attrs.isa<mlir::DictionaryAttr>())
      return failure();
    dict_attr = attrs.cast<mlir::DictionaryAttr>();
  }

  auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
      op, op.call_target_name(), adaptor.getOperands());
  if (dict_attr) {
    NamedAttrList originAttrs = compute_op->getAttrs();
    originAttrs.append(dict_attr);
    compute_op->setAttrs(originAttrs);
  }

  return success();
}
