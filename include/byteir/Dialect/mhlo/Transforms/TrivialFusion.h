//===- TrivialFusion.h ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_TRIVIALFUSION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_TRIVIALFUSION_H

#include "llvm/ADT/DenseMap.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

constexpr StringRef getByteIRTrivialFusionAttrName() {  return "__byteir_trivial_fusion__"; }

void populateTrivialFusionPattern(RewritePatternSet &patterns,
                                  llvm::DenseMap<StringRef, StringRef> &lut_name);

// TODO add more target or list of op in arg
std::unique_ptr<OperationPass<FuncOp>> createTrivialFusionPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_TRIVIALFUSION_H