//===- RemoveFuncBody.h ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_REMOVEFUNCBODY_H
#define BYTEIR_TRANSFORMS_REMOVEFUNCBODY_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

std::unique_ptr<OperationPass<func::FuncOp>>
createRemoveFuncBodyPass(llvm::StringRef anchorTag = "",
                         bool disableForcePrivate = false);

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_REMOVEFUNCBODY_H