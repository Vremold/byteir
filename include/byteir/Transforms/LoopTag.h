//===- LoopTag.h -------------------------------------------------- C++ ---===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_LOOPTAG_H
#define BYTEIR_TRANSFORMS_LOOPTAG_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

std::unique_ptr<OperationPass<func::FuncOp>>
createLoopTagPass(llvm::StringRef anchorTag = "",
                  const std::string &attachTag = "", unsigned depth = 1,
                  const std::string &loopType = "scf.for");

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_LOOPTAG_H
