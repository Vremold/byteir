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

namespace byteir {
struct ArgSideEffectAnalysis;
}

namespace mlir {
class FuncOp;
class ModuleOp;

// Set all memref to a space including intermediate and args
std::unique_ptr<OperationPass<ModuleOp>>
createSetAllSpacePass(const std::string &entryFunc = "",
                      const std::string &space = "",
                      byteir::ArgSideEffectAnalysis *analysis = nullptr);

// Set all args (including return) to a space
std::unique_ptr<OperationPass<ModuleOp>>
createSetArgSpacePass(const std::string &entryFunc = "",
                      const std::string &allSpace = "",
                      bool allowArgWritable = false,
                      byteir::ArgSideEffectAnalysis *analysis = nullptr);

// Set all args and return to a set of specific spaces
std::unique_ptr<OperationPass<ModuleOp>> createSetArgSpacePass(
    const std::string &entryFunc, llvm::ArrayRef<std::string> argSpaces,
    llvm::ArrayRef<std::string> retSpaces, bool allowArgWritable = false,
    byteir::ArgSideEffectAnalysis *analysis = nullptr);

// Set space for all ops
std::unique_ptr<OperationPass<FuncOp>>
createSetOpSpacePass(const std::string &entryFunc = "",
                     const std::string &Space = "");

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_SETSPACE_H
