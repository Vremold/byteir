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

mlir::LogicalResult
mlir::ConvertToByrePattrn<mlir::CallOp>::matchAndRewrite(mlir::CallOp op, typename mlir::CallOp::Adaptor adaptor,
  ConversionPatternRewriter& rewriter) const {

  auto funcOp = GetFuncOp(op);
  if (funcOp == nullptr) {
    return failure();
  }

  StringAttr nameAttr = funcOp->getAttrOfType<StringAttr>(byre::getByreComputeName());

  if (nameAttr == nullptr) {
    return failure();
  }
  
  rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(op,
    nameAttr.getValue(), adaptor.getOperands());

  return success();
}

template<>
mlir::LogicalResult
mlir::ConvertToByrePattrn<mlir::lmhlo::DotOp>::matchAndRewrite(mlir::lmhlo::DotOp op, typename mlir::lmhlo::DotOp::Adaptor adaptor,
  ConversionPatternRewriter& rewriter) const {

  auto found = src_to_callee_.find(op.getOperation()->getName().getStringRef());
  if (found == src_to_callee_.end()) {
    return op->emitOpError() << "can not find matched byre_compute_name";
  }

  auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(op, found->second, adaptor.getOperands());
  // append attribute 'lhs_contracting_dimension' and 'rhs_contracting_dimension'
  auto dot_dimension_numbers = adaptor.dot_dimension_numbers();
  assert(dot_dimension_numbers.getLhsBatchingDimensions().size() == 0);
  assert(dot_dimension_numbers.getRhsBatchingDimensions().size() == 0);
  assert(dot_dimension_numbers.getLhsContractingDimensions().size() == 1);
  assert(dot_dimension_numbers.getRhsContractingDimensions().size() == 1);
  int64_t lhs_contracting_dimension = dot_dimension_numbers.getLhsContractingDimensions()[0];
  int64_t rhs_contracting_dimension = dot_dimension_numbers.getRhsContractingDimensions()[0];

  NamedAttrList attrs(compute_op->getAttrs());
  // TODO: move this outside
  auto lhs_contracting_attr = rewriter.getI64IntegerAttr(lhs_contracting_dimension);
  auto rhs_contracting_attr = rewriter.getI64IntegerAttr(rhs_contracting_dimension);
  attrs.append("lhs_contracting_dimension", lhs_contracting_attr);
  attrs.append("rhs_contracting_dimension", rhs_contracting_attr);
  compute_op->setAttrs(attrs.getDictionary(getContext()));

  return success();
}

//instantiation
template class mlir::ConvertToByrePattrn<mlir::lmhlo::DotOp>;

mlir::LogicalResult
mlir::ConvertToByrePattrn<lmhlo::CustomCallOp>::matchAndRewrite(
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
