//===- TrivialFusion.h ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_TRIVIALFUSION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_TRIVIALFUSION_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

constexpr StringRef getByteIRTrivialFusionAttrName() {
  return "__byteir_trivial_fusion__";
}

void populateTrivialFusionPattern(
    RewritePatternSet &patterns,
    llvm::DenseMap<StringRef, StringRef> &lut_name);

// TODO add more target or list of op in arg
std::unique_ptr<OperationPass<func::FuncOp>> createTrivialFusionPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_TRIVIALFUSION_H