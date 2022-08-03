//===- ArithToMhlo.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_TOMHLO_ARITHTOMHLO_H
#define BYTEIR_CONVERSION_TOMHLO_ARITHTOMHLO_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

std::unique_ptr<OperationPass<func::FuncOp>> createConvertArithToMhloPass();

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOMHLO_ARITHTOMHLO_H