//===- LoopUnroll.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_LOOPUNROLL_H
#define BYTEIR_TRANSFORMS_LOOPUNROLL_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

constexpr StringRef getByteIRUnorllAttrName() { return "__byteir_unroll__"; }

std::unique_ptr<OperationPass<FuncOp>> 
createByteIRLoopUnrollPass(unsigned factor = 2, bool upTo = false, bool full = false, int depth = -1);

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_LOOPUNROLL_H