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
      : OpConversionPattern<SrcOpTy>(ctx), srcToCallee(lut),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(SrcOpTy op, typename SrcOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto found = srcToCallee.find(op.getOperation()->getName().getStringRef());
    if (found == srcToCallee.end()) {
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
  const llvm::DenseMap<StringRef, StringRef> &srcToCallee;
  bool appendArgTypes;
};

template <typename SrcOpTy>
class ConvertToByrePatternWithAllAttrs : public OpConversionPattern<SrcOpTy> {
public:
  ConvertToByrePatternWithAllAttrs(
      MLIRContext *ctx, const llvm::DenseMap<StringRef, StringRef> &lut,
      bool appendTypes)
      : OpConversionPattern<SrcOpTy>(ctx), srcToCallee(lut),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(SrcOpTy op, typename SrcOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto found = srcToCallee.find(op.getOperation()->getName().getStringRef());
    if (found == srcToCallee.end()) {
      // TODO adding more error message
      return failure();
    }

    auto key = getByreKey(found->second, op->getOperandTypes(), appendArgTypes);

    auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
        op, key, adaptor.getOperands());
    addAttrs(compute_op.getOperation(), op->getAttrs());
    return success();
  }

protected:
  const llvm::DenseMap<StringRef, StringRef> &srcToCallee;
  bool appendArgTypes;
};

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOBYRE_COMMON_H