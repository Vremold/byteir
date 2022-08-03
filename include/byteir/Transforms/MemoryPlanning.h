//===- MemoryPlanning.h ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_MEMORY_PLANNING_H
#define BYTEIR_TRANSFORMS_MEMORY_PLANNING_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

std::unique_ptr<OperationPass<func::FuncOp>> createMemoryPlanningPass();

/// couldReuseBuffer is a user provided callback which receives a Value as
/// parameter and returns whether the allocation corresponding to the Value can
/// be reused
std::unique_ptr<OperationPass<func::FuncOp>>
createMemoryPlanningPass(std::function<bool(Value)> couldReuseAllocation);

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_MEMORY_PLANNING_H