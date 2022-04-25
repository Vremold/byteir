//===- ToGPU.h ----------------------------------------------------- C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_TOGPU_H
#define BYTEIR_CONVERSION_TOGPU_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {

// Note: this pass is soft-deprecated
// Please use FuncTag, LoopTag and ConvertFuncToGPUPass instead
std::unique_ptr<OperationPass<FuncOp>>
createCoalescedForToGPULaunchPass(int64_t bSize = 32);

// some constexpr attr name
// TODO add a namespace if conflict

// attr to label a function to gpu::GPUFuncOp
constexpr StringRef getToGPUAttrName() { return "__byteir_to_gpu__"; }

// attr to label a loop to gpu simt
constexpr StringRef getLoopToSIMTAttrName() {
  return "__byteir_loop_to_simt__";
}

// attr to label a loop to enable coarsening
constexpr StringRef getCoarsenSIMTAttrName() {
  return "__byteir_coarsen_simt__";
}

// attr to label a loop with id x, y, z
constexpr StringRef getLinearIdXName() { return "linear_id.x"; }
constexpr StringRef getLinearIdYName() { return "linear_id.y"; }
constexpr StringRef getLinearIdZName() { return "linear_id.z"; }
constexpr StringRef getThreadIdXName() { return "thread_id.x"; }
constexpr StringRef getThreadIdYName() { return "thread_id.y"; }
constexpr StringRef getThreadIdZName() { return "thread_id.z"; }
constexpr StringRef getBlockIdXName() { return "block_id.x"; }
constexpr StringRef getBlockIdYName() { return "block_id.y"; }
constexpr StringRef getBlockIdZName() { return "block_id.z"; }
constexpr StringRef getGridIdXName() { return "grid_id.x"; }
constexpr StringRef getGridIdYName() { return "grid_id.y"; }
constexpr StringRef getGridIdZName() { return "grid_id.z"; }

std::unique_ptr<OperationPass<ModuleOp>>
createConvertFuncToGPUPass(ArrayRef<int64_t> bs = {32, 1, 1},
                           ArrayRef<int64_t> gs = {32, 1, 1},
                           const std::string &moduleName = "unified");

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOGPU_H
