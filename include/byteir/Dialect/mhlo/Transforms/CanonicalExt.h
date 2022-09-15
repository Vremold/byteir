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
class ConcatenateOp;
class DynamicBroadcastInDimOp;
class DynamicConvOp;

// Most of these will push back to upstream
// So this file only includes patterns, not a pass.

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
LogicalResult foldBroadcastInDim(BroadcastInDimOp op,
                                 PatternRewriter &rewriter);

///
///  Fold concatenate of continuous slices
///
LogicalResult foldConcatWithContinuousSlices(mhlo::ConcatenateOp op,
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
LogicalResult foldShapeBroadcast(shape::BroadcastOp op,
                                 PatternRewriter &rewriter);

// fold binary op with large constant op
template <typename Op, template <typename> typename Func>
LogicalResult foldLargeBinaryOp(Op op, PatternRewriter &rewriter);

// mhlo.dynamic_conv => mhlo.convolution canonicalization
LogicalResult simplifyDynamicConvToConv(mhlo::DynamicConvOp op,
                                        PatternRewriter &rewriter);

// constant folding for mhlo.concatenate with large result
LogicalResult foldLargeConcatenate(mhlo::ConcatenateOp op,
                                   PatternRewriter &rewriter);

LogicalResult foldTransposeNonSplat(mhlo::TransposeOp op,
                                    PatternRewriter &rewriter);

// populate canonicalizeExt patterns
void populateCanonicalizeExtPatterns(RewritePatternSet &patterns);

// Get all canonicalizationExt on top of canoncialization
void getCanonicalizationExtPatterns(RewritePatternSet &results,
                                    MLIRContext *context);

} // namespace mhlo
} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_CANONICALEXT_H
