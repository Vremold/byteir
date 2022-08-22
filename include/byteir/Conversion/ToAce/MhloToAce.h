//===- MhloToAce.h --------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_TOACE_MHLOTOACE_H
#define BYTEIR_CONVERSION_TOACE_MHLOTOACE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

std::unique_ptr<OperationPass<func::FuncOp>> createConvertMhloToAcePass();

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOACE_MHLOTOACE_H