//===- Passes.h -----------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TARGET_PTX_PASSES_H
#define BYTEIR_TARGET_PTX_PASSES_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir {

namespace gpu {
  class GPUModuleOp;
}

/// Convert kernel functions in GPU dialect to PTX
std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
  createSerializeToPTXPass(unsigned optLevel, const std::string& libdeviceFile, 
                            const std::string& triple, const std::string& targetChip, 
                            const std::string& features, std::string& targetISA);

} // namespace mlir

#endif // BYTEIR_TARGET_PTX_PASSES_H