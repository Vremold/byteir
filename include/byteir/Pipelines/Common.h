//===- Common.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_COMMON_H
#define BYTEIR_PIPELINES_COMMON_H

#include "mlir/Pass/PassManager.h"

namespace mlir {

void addCleanUpPassPipeline(OpPassManager& pm);

void addMultiCSEPipeline(OpPassManager& pm, unsigned cnt);

} // namespace mlir

#endif // BYTEIR_PIPELINES_COMMON_H