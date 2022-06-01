//===- SetSpace.h ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_SETSPACE_H
#define BYTEIR_TRANSFORMS_SETSPACE_H

#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>
#include <string>

namespace mlir {

// Set all memref to a space including intermediate and args
std::unique_ptr<OperationPass<ModuleOp>>
createSetAllSpacePass(std::string entryFunc = "",
                      const std::string &space = "");

// Set all args (including return) to a space
std::unique_ptr<OperationPass<ModuleOp>>
createSetArgSpacePass(std::string entryFunc = "",
                      const std::string &allSpace = "",
                      bool allowArgWritable = false);

// Set all args and return to a set of specific spaces
std::unique_ptr<OperationPass<ModuleOp>> createSetArgSpacePass(
    std::string entryFunc, llvm::ArrayRef<std::string> argSpaces,
    llvm::ArrayRef<std::string> retSpaces, bool allowArgWritable = false);

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_SETSPACE_H
