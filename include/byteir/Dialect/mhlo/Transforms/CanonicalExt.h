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

namespace shape {
class BroadcastOp;
}

namespace mhlo {
class TransposeOp;
class BroadcastInDimOp;
class DynamicBroadcastInDimOp;

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

// fold the pattern like this
// const0 = [4]
// c = mhlo.xxx(a, b)
// d = shape.shape_of(c)
// e = shape.broadcast(d, const0)
//
// remove the shape.broadcast
//
// note the shape of c is [...., 4], ... means any, include dynamic shape
//
// generally, when const0 = c.shape[any:], this pattern would fold successfully
LogicalResult FoldShapeBroadcast(shape::BroadcastOp op,
                                 PatternRewriter &rewriter);

// Get all canoncializationExt on top of canoncialization
void getCanonicalizationExtPatterns(RewritePatternSet &results,
                                    MLIRContext *context);

} // namespace mhlo
} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_CANONICALEXT_H
