//===- UnregisteredToAce.h ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_TOACE_UNREGISTEREDTOACE_H
#define BYTEIR_CONVERSION_TOACE_UNREGISTEREDTOACE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertUnregisteredToAcePass();

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOACE_UNREGISTEREDTOACE_H