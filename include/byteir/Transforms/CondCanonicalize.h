//===- CondCanonicalize.h ------------------------------------------ C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_CONDCANONICALIZE_H
#define BYTEIR_TRANSFORMS_CONDCANONICALIZE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class RewritePatternSet;
namespace func {
class FuncOp;
} // namespace func

void populateCondCanonicalizePatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<func::FuncOp>> createCondCanonicalizePass();

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_CONDCANONICALIZE_H
