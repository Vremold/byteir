//===- Tiling.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_TILING_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_TILING_H

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

// tiling attr names
constexpr StringRef getScopeTilingAnchorAttrName() {
  return "__byteir_scope_tile_anchor__";
}

// TODO change the following to a struct attr
constexpr StringRef getScopeTilingAxisAttrName() {
  return "__byteir_scope_tile_axis__";
}

constexpr StringRef getScopeTilingRankAttrName() {
  return "__byteir_scope_tile_rank__";
}

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
