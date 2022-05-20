//===- GenericDeviceConfig.h ----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_GENERICDEVICECONFIG_H
#define BYTEIR_TRANSFORMS_GENERICDEVICECONFIG_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

std::unique_ptr<OperationPass<FuncOp>>
createGenericDeviceConfigPass(llvm::StringRef anchorTag = "",
                              llvm::StringRef computeName = "");

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_GENERICDEVICECONFIG_H