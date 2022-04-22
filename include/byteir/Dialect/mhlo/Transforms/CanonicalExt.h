//===- CanonicalExt.h ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_CANONICALEXT_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_CANONICALEXT_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class MLIRContext;

namespace mhlo {
class TransposeOp;
class BroadcastInDimOp;

// Most of these will push back to upstream
// So this file only includes patterns, not a pass.

///
///  Transpose
///
LogicalResult EliminateSplatConstantTranspose(mhlo::TransposeOp op,
                                              PatternRewriter &rewriter);

///
///  BroadcastInDim
///
/// BroadcastInDim could be folded in some special cases. Ex.
///
/// const
///   \
///   broadcast_in_dim  const
///       \              /
///             mul
LogicalResult FoldBroadcastInDim(BroadcastInDimOp op,
                                 PatternRewriter &rewriter);

// Get all canoncializationExt on top of canoncialization
void getCanonicalizationExtPatterns(RewritePatternSet &results,
                                    MLIRContext *context);

} // namespace mhlo
} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_CANONICALEXT_H