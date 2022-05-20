//===- SetArgSpace.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_SETARGSPACE_H
#define BYTEIR_TRANSFORMS_SETARGSPACE_H

#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>
#include <string>

namespace mlir {

std::unique_ptr<OperationPass<ModuleOp>>
createSetArgSpacePass(std::string entryFunc = "", std::string allSpace = "");

std::unique_ptr<OperationPass<ModuleOp>>
createSetArgSpacePass(std::string entryFunc,
                      llvm::ArrayRef<std::string> spaces);

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_SETARGSPACE_H
