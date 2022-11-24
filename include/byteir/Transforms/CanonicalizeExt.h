//===- CanonicalizeExt.h --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_CANONICALIZEEXT_H
#define BYTEIR_TRANSFORMS_CANONICALIZEEXT_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

/// Creates an instance of the CanonicalizeExt pass, configured with default
/// settings (which can be overridden by pass options on the command line).
std::unique_ptr<Pass> createCanonicalizeExtPass();

/// Creates an instance of the CanonicalizeExt pass with the specified config.
std::unique_ptr<Pass>
createCanonicalizeExtPass(const GreedyRewriteConfig &config,
                          ArrayRef<std::string> disabledPatterns = llvm::None,
                          ArrayRef<std::string> enabledPatterns = llvm::None);

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_CANONICALIZEEXT_H
