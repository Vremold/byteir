//===- Common.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_TOBYRE_COMMON_H
#define BYTEIR_CONVERSION_TOBYRE_COMMON_H

#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {

std::string getByreKey(StringRef original, TypeRange types,
                       bool appendArgTypes);

template <typename SrcOpTy>
class ConvertToByrePattern : public OpConversionPattern<SrcOpTy> {
public:
  ConvertToByrePattern(MLIRContext *ctx,
                       const llvm::DenseMap<StringRef, StringRef> &lut,
                       bool appendTypes)
      : OpConversionPattern<SrcOpTy>(ctx), src_to_callee_(lut),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(SrcOpTy op, typename SrcOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto found =
        src_to_callee_.find(op.getOperation()->getName().getStringRef());
    if (found == src_to_callee_.end()) {
      // TODO adding more error message
      return failure();
    }

    auto key = getByreKey(found->second, op->getOperandTypes(), appendArgTypes);

    // Note all attrs will be removed
    rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(op, key,
                                                       adaptor.getOperands());

    return success();
  }

protected:
  const llvm::DenseMap<StringRef, StringRef> &src_to_callee_;
  bool appendArgTypes;
};

template <typename SrcOpTy>
class ConvertToByrePatternWithAllAttrs : public OpConversionPattern<SrcOpTy> {
public:
  ConvertToByrePatternWithAllAttrs(
      MLIRContext *ctx, const llvm::DenseMap<StringRef, StringRef> &lut,
      bool appendTypes)
      : OpConversionPattern<SrcOpTy>(ctx), src_to_callee_(lut),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(SrcOpTy op, typename SrcOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto found =
        src_to_callee_.find(op.getOperation()->getName().getStringRef());
    if (found == src_to_callee_.end()) {
      // TODO adding more error message
      return failure();
    }

    auto key = getByreKey(found->second, op->getOperandTypes(), appendArgTypes);

    auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
        op, key, adaptor.getOperands());
    AddAttrs(compute_op.getOperation(), op->getAttrs());
    return success();
  }

protected:
  const llvm::DenseMap<StringRef, StringRef> &src_to_callee_;
  bool appendArgTypes;
};

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOBYRE_COMMON_H