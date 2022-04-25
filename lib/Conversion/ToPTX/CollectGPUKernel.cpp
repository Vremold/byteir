//===- CollectGPUKernel.cpp -----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "byteir/Conversion/ToPTX/ToPTX.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"
#include <iostream>

using namespace mlir;
using namespace mlir::byre;
using namespace mlir::gpu;
using namespace llvm;

namespace {

// Main Pass
struct CollectGPUKernelPass
    : public CollectGPUKernelBase<CollectGPUKernelPass> {

  CollectGPUKernelPass(const std::string &name) : CollectGPUKernelBase() {
    this->moduleName = name;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    SmallVector<gpu::GPUModuleOp> gmCollector;
    SmallVector<Operation *> removeOps;
    bool found = false;
    GPUModuleOp dst;

    for (auto &op : m.getBody()->without_terminator()) {
      if (auto gm = dyn_cast<gpu::GPUModuleOp>(op)) {
        if (gm.getName() == moduleName) {
          found = true;
          dst = gm;
        } else {
          gmCollector.push_back(gm);
        }
      }
    }

    // Note FuncOps not in m.getBody()->without_terminator()
    for (auto func : m.getOps<FuncOp>()) {
      removeOps.push_back(func);
    }

    if (gmCollector.size() == 0) {
      for (auto op : removeOps) {
        op->erase();
      }
      return;
    }

    if (!found) {
      OpBuilder builder = OpBuilder::atBlockBegin(m.getBody());
      dst = builder.create<GPUModuleOp>(m.getLoc(), moduleName);
    }

    SymbolTable dstTable(dst);

    for (auto gm : gmCollector) {
      for (auto &op : gm.getBody()->without_terminator()) {
        auto newOp = op.clone();
        dstTable.insert(newOp);
      }
      gm.erase();
    }

    for (auto op : removeOps) {
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createCollectGPUKernelPass(const std::string &name) {
  return std::make_unique<CollectGPUKernelPass>(name);
}
