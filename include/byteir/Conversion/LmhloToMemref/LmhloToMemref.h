//===- LmhloToMemref.h ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_LMHLOTOMEMREF_H
#define BYTEIR_CONVERSION_LMHLOTOMEMREF_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

void populateLmhloToMemrefPattern(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<func::FuncOp>> createLmhloToMemrefPass();

} // namespace mlir

#endif // BYTEIR_CONVERSION_LMHLOTOMEMREF_H
