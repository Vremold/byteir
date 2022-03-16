//===- CanonicalExt.cpp ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/CanonicalExt.h"
#include "byteir/Utils/AttrUtils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include <iostream> 

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

LogicalResult
mlir::mhlo::EliminateSplatConstantTranspose(mhlo::TransposeOp op,
                                            PatternRewriter &rewriter) {

  if (!op.getType().hasStaticShape()) {
    return failure();
  }

  auto const_op = op.operand().getDefiningOp<mhlo::ConstOp>();
  if (!const_op) {
    return failure();
  }

  auto maybe_new_attr = reshapeSplatElementsAttr(const_op.value(), op.getType());
  if (!maybe_new_attr.hasValue()) return failure();

  rewriter.replaceOpWithNewOp<mhlo::ConstOp>(op, maybe_new_attr.getValue());
  return success();
}


void mlir::mhlo::getCanonicalizationExtPatterns(RewritePatternSet &results,
                                                MLIRContext *ctx) {

  // add dialect level getCanonicalizationPatterns
  auto mhlo_dailect = ctx->getOrLoadDialect<mhlo::MhloDialect>();
  if (mhlo_dailect) {
    mhlo_dailect->getCanonicalizationPatterns(results);
  }

  // add op level  getCanonicalizationPatterns
  for (RegisteredOperationName op : ctx->getRegisteredOperations()) {
    // only add mhlo-related 
    if (isa<MhloDialect>(op.getDialect())) {
      op.getCanonicalizationPatterns(results, ctx);
    }
  }

  // add our extension 
  results.add(mlir::mhlo::EliminateSplatConstantTranspose);
}
