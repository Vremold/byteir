//===- CMAE.h -------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_CMAE_H
#define BYTEIR_TRANSFORMS_CMAE_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

// CMAE pass
std::unique_ptr<OperationPass<func::FuncOp>>
createCMAEPass(const std::string &skip = "");

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_CMAE_H
