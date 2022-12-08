//===- Tiling.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_TILING_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_TILING_H

#include "byteir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace linalg_ext {

constexpr StringRef getLinalgExtTileAttrName() { return "__byteir_tile__"; }

constexpr StringRef getLinalgExtTileAndFuseAttrName() {
  return "__byteir_tile_and_fuse_";
}

/// Base rewrite pattern to tile and distribute operations that implement the
/// `TiledOpInterface`.
/// Base pattern for tiling TiledOpInterfaceOps.
struct TilingInterfaceBaseTilingPattern
    : public OpInterfaceRewritePattern<TilingInterface> {
  TilingInterfaceBaseTilingPattern(
      MLIRContext *context, scf::SCFTilingOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(context, benefit), filter(filter),
        options(options) {}

  LogicalResult matchAndRewriteBase(TilingInterface tilableOp,
                                    PatternRewriter &rewriter,
                                    scf::SCFTilingResult &result) const;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// Options to control tiling;
  scf::SCFTilingOptions options;
};

struct TilingInterfaceTilingPattern : public TilingInterfaceBaseTilingPattern {
  TilingInterfaceTilingPattern(
      MLIRContext *context, scf::SCFTilingOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : TilingInterfaceBaseTilingPattern(context, options, filter, benefit) {}

  LogicalResult matchAndRewrite(TilingInterface tilableOp,
                                PatternRewriter &rewriter) const;
};
} // namespace linalg_ext

std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgOpTilingPass(ArrayRef<int64_t> tileSizes = {},
                         linalg::LinalgTilingLoopType loopType =
                             linalg::LinalgTilingLoopType::Loops);

std::unique_ptr<OperationPass<func::FuncOp>> createLinalgScopeTilingPass(
    int64_t tileAxis = 0, int64_t tileSize = 0,
    bool parallelizeReduction = false,
    linalg::LinalgTilingLoopType loopType = linalg::LinalgTilingLoopType::Loops,
    bool keepTag = false);

} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_TILING_H
