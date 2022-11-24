//===- Transforms.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir {
namespace linalg_ext {

// LinalgTransforms and LinalgTransformationFilter will be deprecated soon
struct LinalgTransforms {
  static const StringLiteral kLinalgTransformMarker;
};

/// Helper class to control application of linalg transformation patterns.
/// Control comes in 2 forms:
///   1. attribute matching and setting behavior using the attribute named
///      `kLinalgTransformMarker`. This can be used to build a state machine
///      using attributes and incrementally applying patterns to advance states.
///   2. filter function, which is a simple lambda on the Operation* that
///      returns a LogicalResult.
struct LinalgTransformationFilter {
  using FilterFunction = std::function<LogicalResult(Operation *)>;

  explicit LinalgTransformationFilter(
      ArrayRef<StringAttr> matchDisjunction = {},
      Optional<StringAttr> replacement = None);

  explicit LinalgTransformationFilter(
      const FilterFunction &f, ArrayRef<StringAttr> matchDisjunction = {},
      Optional<StringAttr> replacement = None);

  LinalgTransformationFilter(LinalgTransformationFilter &&) = default;
  LinalgTransformationFilter(const LinalgTransformationFilter &) = default;
  LogicalResult checkAndNotify(PatternRewriter &rewriter, Operation *op) const;
  void replaceLinalgTransformationFilter(PatternRewriter &rewriter,
                                         Operation *op) const;
  bool hasReplacementFilter(Operation *op) const;

  LinalgTransformationFilter &addFilter(const FilterFunction &f) {
    if (f)
      filters.push_back(f);
    return *this;
  }

  template <typename... OpTypes> LinalgTransformationFilter &addOpFilter() {
    return addFilter(
        [](Operation *op) { return success(isa<OpTypes...>(op)); });
  }

  LinalgTransformationFilter &addOpNameFilter(StringRef opName) {
    return addFilter([opName](Operation *op) {
      return success(op->getName().getStringRef() == opName);
    });
  }

  LinalgTransformationFilter &setMatchByDefault() {
    matchByDefault = true;
    return *this;
  }

private:
  SmallVector<FilterFunction> filters;
  SmallVector<StringAttr> matchDisjunction;
  Optional<StringAttr> replacement;
  /// When set to true, if the attribute is not set, it will be treated as
  /// a match. Default is false.
  bool matchByDefault;
};

/// Structure to represent the result of tiling operation.
struct TiledOp {
  /// Tiled op.
  Operation *op;
  /// Loops generated during tiling.
  SmallVector<Operation *> loops;
  /// Values that are replacements for the untiled operations.
  SmallVector<Value> results;
};

/// Main entry point for tiling LinalgExtOps using TiledOpInterface.
FailureOr<TiledOp> tileLinalgExtOp(RewriterBase &b, TilingInterface tilableOp,
                                   const linalg::LinalgTilingOptions &options);

/// Base rewrite pattern to tile and distribute operations that implement the
/// `TiledOpInterface`.
/// Base pattern for tiling TiledOpInterfaceOps.
struct TilingInterfaceBaseTilingPattern
    : public OpInterfaceRewritePattern<TilingInterface> {
  TilingInterfaceBaseTilingPattern(
      MLIRContext *context, linalg::LinalgTilingOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(context, benefit), filter(filter),
        options(options) {}

  LogicalResult matchAndRewriteBase(TilingInterface tilableOp,
                                    PatternRewriter &rewriter,
                                    TiledOp &result) const;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// Options to control tiling;
  linalg::LinalgTilingOptions options;
};

struct TilingInterfaceTilingPattern : public TilingInterfaceBaseTilingPattern {
  TilingInterfaceTilingPattern(
      MLIRContext *context, linalg::LinalgTilingOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : TilingInterfaceBaseTilingPattern(context, options, filter, benefit) {}

  LogicalResult matchAndRewrite(TilingInterface tilableOp,
                                PatternRewriter &rewriter) const;
};

} // namespace linalg_ext
} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H