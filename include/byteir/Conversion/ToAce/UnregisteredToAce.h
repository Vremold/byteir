//===- UnregisteredToAce.h ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_TOACE_UNREGISTEREDTOACE_H
#define BYTEIR_CONVERSION_TOACE_UNREGISTEREDTOACE_H

#include "byteir/Dialect/Ace/AceDialect.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

std::unique_ptr<OperationPass<FuncOp>> createConvertUnregisteredToAcePass();

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOACE_UNREGISTEREDTOACE_H