//===- ReduceFusion.h ------------------------------------------*--- C++
//-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_REDUCEFUSION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_REDUCEFUSION_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

constexpr StringRef getByteIRReduceFusionAttrName() {
  return "__byteir_reduce_fusion__";
}

// fuse ReduceWindow with Pad and/or Constant
void populateFuseReduceWindowPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<func::FuncOp>> createReduceFusionPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_REDUCEFUSION_H