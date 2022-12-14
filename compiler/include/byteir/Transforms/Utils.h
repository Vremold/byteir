//===- Utils.h ------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_UTILS_H
#define BYTEIR_TRANSFORMS_UTILS_H

#include "llvm/ADT/StringRef.h"
#include <functional>

// This file includes all RewritePattern-form of utils.
// It is similiar to ones in byteir/Utils but in RewritePattern.
namespace mlir {
class DominanceInfo;
class Operation;
class PostDominanceInfo;
class RewritePatternSet;

void populateHoistUpInBlockPatterns(
    RewritePatternSet &patterns, DominanceInfo &domInfo,
    const std::function<bool(Operation *)> &checkFunc);

void populateHoistDownInBlockPatterns(
    RewritePatternSet &patterns, PostDominanceInfo &postDomInfo,
    const std::function<bool(Operation *)> &checkFunc);

void populateRemoveAttrPatterns(RewritePatternSet &patterns,
                                llvm::StringRef attrName);

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_UTILS_H
