//===- AffineToGPU.h ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_AFFINETOGPU_H
#define BYTEIR_CONVERSION_AFFINETOGPU_H

#include "mlir/Pass/Pass.h"
#include <memory>
//#include <string>

namespace mlir {

std::unique_ptr<OperationPass<FuncOp>>
createCoalescedForToGPULaunchPass(int64_t bSize = 32);

} // namespace mlir

#endif // BYTEIR_CONVERSION_AFFINETOGPU_H