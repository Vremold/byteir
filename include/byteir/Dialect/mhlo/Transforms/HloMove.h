//===- HloMove.h ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_HLOMOVE_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_HLOMOVE_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
class RewritePatternSet;
namespace func {
class FuncOp;
} // namespace func

// Note MoveDown and MoveUp are mutual exclusive
// in an applyPatternsAndFoldGreedily pass.
// However, they can still run together in different passes in a pipeline.

void populateHloMoveDownPattern(
    RewritePatternSet &patterns,
    const llvm::DenseSet<llvm::StringRef> &blocker = {},
    bool allMultiUser = false, bool multiUser = false);

void populateHloMoveUpPattern(
    RewritePatternSet &patterns,
    const llvm::DenseSet<llvm::StringRef> &blocker = {},
    bool multiInput = false);

// TODO add more target or list of op in arg
std::unique_ptr<OperationPass<func::FuncOp>>
createHloMoveDownPass(bool allMultiUser = false, bool multiUser = false);

std::unique_ptr<OperationPass<func::FuncOp>>
createHloMoveUpPass(bool multiInput = false);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_HLOMOVE_H