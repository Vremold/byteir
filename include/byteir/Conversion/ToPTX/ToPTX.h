//===- ToPTX.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_TOPTX_H
#define BYTEIR_CONVERSION_TOPTX_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

std::unique_ptr<FunctionPass> createGenPTXConfigPass();


} // namespace mlir

#endif // BYTEIR_CONVERSION_TOPTX_H