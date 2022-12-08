//===- Transforms.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace scf {
/// tileConsumerAndFuseProducerUsingSCFForOpExt is an enhanced version
/// tileConsumerAndFuseProducerGreedilyUsingSCFForOp.
FailureOr<scf::SCFTileAndFuseResult>
tileConsumerAndFuseProducerUsingSCFForOpExt(
    RewriterBase &rewriter, TilingInterface consumer,
    const scf::SCFTileAndFuseOptions &options);

void labelTileLoopType(Operation *op, ArrayRef<scf::ForOp> loops);

LogicalResult isValidTiling(Operation *tiled);
} // namespace scf

namespace linalg_ext {

/// return a list of utils::IteratorType for a given op
/// and list of scf::ForOp loops
///
/// ```mlir
/// Example 1:
/// scf.for %iv_m      // m_loop
///   scf.for %iv_k    // k_loop
///     scf.for %iv_n  // n_loop
///       extract_slice_A
///       extract_slice_B
///       extract_slice_C
///       %0 = linalg.matmul ins (extract_slice_A, extract_slice_B)
///                          outs(extract_slice_C)
/// ```
/// loops = [m_loop, k_loop, n_loop], op = linalg.matmul
/// return [parallel, reduction, parallel]
///
FailureOr<llvm::SmallVector<llvm::Optional<utils::IteratorType>>>
getLoopIteratorTypes(Operation *op, ArrayRef<scf::ForOp> loops);

void mergeLoopIteratorTypes(
    llvm::SmallVector<llvm::Optional<utils::IteratorType>> &from,
    llvm::SmallVector<llvm::Optional<utils::IteratorType>> &to);

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

} // namespace linalg_ext
} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H