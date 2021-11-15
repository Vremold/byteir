//===- Common.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_CONVERTTOBYRE_COMMON_H
#define BYTEIR_CONVERSION_CONVERTTOBYRE_COMMON_H

#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
template <typename SrcOpTy>
class ConvertToByrePattern : public OpConversionPattern<SrcOpTy> {
 public:

  ConvertToByrePattern(MLIRContext* ctx,
    const llvm::DenseMap<StringRef, StringRef>& lut)
    : OpConversionPattern<SrcOpTy>(ctx), src_to_callee_(lut) { }

  LogicalResult
    matchAndRewrite(SrcOpTy op, typename SrcOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {

    auto found = src_to_callee_.find(op.getOperation()->getName().getStringRef());
    if (found == src_to_callee_.end()) {
      // TODO adding more error message
      return failure();
    }

    // Note all attrs will be removed
    rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(op, 
      found->second, adaptor.getOperands());

    return success();
  }

private:
  const llvm::DenseMap<StringRef, StringRef>& src_to_callee_;
};

template<> 
class ConvertToByrePattern<mlir::CallOp> : public OpConversionPattern<mlir::CallOp> {
public:
  ConvertToByrePattern(MLIRContext* ctx)
    : OpConversionPattern<mlir::CallOp>(ctx) {}

  LogicalResult
    matchAndRewrite(mlir::CallOp op, typename mlir::CallOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};

template <>
class ConvertToByrePattern<mlir::lmhlo::DotOp> : public OpConversionPattern<mlir::lmhlo::DotOp> {
public:
  ConvertToByrePattern(MLIRContext* ctx)
    : OpConversionPattern<mlir::lmhlo::DotOp>(ctx) {}
  
  LogicalResult
    matchAndRewrite(mlir::lmhlo::DotOp op, typename mlir::lmhlo::DotOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};

template <>
class ConvertToByrePattern<lmhlo::CustomCallOp>
    : public OpConversionPattern<lmhlo::CustomCallOp> {
public:
  ConvertToByrePattern(MLIRContext *ctx)
      : OpConversionPattern<lmhlo::CustomCallOp>(ctx) {}

  LogicalResult
  matchAndRewrite(lmhlo::CustomCallOp op,
                  typename lmhlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace mlir

#endif // BYTEIR_CONVERSION_CONVERTTOBYRE_COMMON_H