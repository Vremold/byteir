//===- ToByre.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_TOBYRE_H
#define BYTEIR_CONVERSION_TOBYRE_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
// forward decl
class RewritePatternSet;
class ModuleOp;
namespace func {
class FuncOp;
} // namespace func

// Collect a set of patterns to convert ops from Lmhlo dialect to Byre dialect
// Note: supportMap is a reference.
void populateLmhloToByreConversionPatterns(
    RewritePatternSet &patterns, llvm::StringMap<StringRef> &supportMap,
    bool appendArgTypes);

void populateViewLikeToByreConversionPatterns(RewritePatternSet &patterns);

void populateStdToByreConversionPatterns(RewritePatternSet &patterns,
                                         bool appendArgTypes);

// Collect a set of patterns to convert ops from Ace dialect to Byre dialect
// void populateAceToByreConversionPatterns(RewritePatternSet& patterns);
std::unique_ptr<OperationPass<ModuleOp>>
createConvertToByrePass(bool appendArgTypes = false);

std::unique_ptr<OperationPass<ModuleOp>>
createConvertFuncAndCallToByrePass(bool appendArgTypes = false);

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertLmhloToByrePass(bool appendArgTypes = false);

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOBYRE_H
