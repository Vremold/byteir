//===- TorchXLA.h ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTFUNCTOCUSTOMCALL_TORCH_XLA_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTFUNCTOCUSTOMCALL_TORCH_XLA_H

#include "mlir/Pass/Pass.h"
#include <memory>

// tentative
// FIXME: remove the entire file and folder after TorchXLA pipeline settled down

namespace mlir {
class ModuleOp;

std::unique_ptr<OperationPass<ModuleOp>>
createConvertFuncToCustomCallTorchXLAPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTFUNCTOCUSTOMCALL_TORCH_XLA_H