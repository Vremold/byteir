//===- ToByre.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_CONVERTTOBYRE_H
#define BYTEIR_CONVERSION_CONVERTTOBYRE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
// forward decl
class RewritePatternSet;

// Collect a set of patterns to convert ops from Lmhlo dialect to Byre dialect
// Note: supportMap is a reference.
void populateLmhloToByreConversionPatterns(RewritePatternSet& patterns, 
  llvm::DenseMap<StringRef, StringRef>& supportMap, 
  bool appendArgTypes);


void populateStdToByreConversionPatterns(RewritePatternSet& patterns);

// Collect a set of patterns to convert ops from Ace dialect to Byre dialect
//void populateAceToByreConversionPatterns(RewritePatternSet& patterns);

std::unique_ptr<OperationPass<ModuleOp>> createConvertToByrePass(bool appendArgTypes = false);


} // namespace mlir

#endif // BYTEIR_CONVERSION_CONVERTTOBYRE_H