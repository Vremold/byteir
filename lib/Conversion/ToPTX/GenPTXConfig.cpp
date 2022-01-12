//===- GenPTXConfig.cpp ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "byteir/Analysis/Alias.h"
#include "byteir/Conversion/ToByre/Common.h"
#include "byteir/Conversion/ToPTX/ToPTX.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include <functional>

using namespace byteir;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::byre;
using namespace mlir::gpu;
using namespace mlir::memref;
using namespace llvm;

namespace {

bool IsAliasOp(Operation& op) {
  return isa<memref::CollapseShapeOp>(op) ||
    isa<memref::ExpandShapeOp>(op) ||
    isa<memref::ReshapeOp>(op);
};

// support static for now
// TODO extend it to support dynamic block/grid sizes
// TODO unify CUDA/PTX into the same pass with compilation option

static void AddFuncAttrs(FuncOp func) {
  // handle elementwise fusion
  if (func->hasAttr(getByreElementwiseFusionName())) {
    mlir::OpBuilder opBuilder(func);

    if (func.getOps<gpu::LaunchFuncOp>().empty()) return;

    gpu::LaunchFuncOp launchOp = *func.getOps<gpu::LaunchFuncOp>().begin();

    func->setAttr(getByrePrefix() + "kernel_name",
      opBuilder.getStringAttr(launchOp.getKernelName().getValue()));

    // Handle 1D only, since element-wise is only using 1D (linearized)
    auto grid = launchOp.getGridSizeOperandValues();
    int64_t gx = cast<ConstantIndexOp>(grid.x.getDefiningOp()).value();
    func->setAttr(getByrePrefix() + "GridSize.x",
      opBuilder.getIntegerAttr(opBuilder.getIntegerType(32), gx));

    auto block = launchOp.getBlockSizeOperandValues();
    int64_t bx = cast<ConstantIndexOp>(block.x.getDefiningOp()).value();
    func->setAttr(getByrePrefix() + "BlockSize.x",
      opBuilder.getIntegerAttr(opBuilder.getIntegerType(32), bx));

    func->setAttr(getByreComputeName(), opBuilder.getStringAttr("PTXOp"));

    // Handle arg mapping here 
    // LWC: this is tentative when we are using GPU Kernel Outlining.
    // TODO: drop this when we are arrange our arg placement in our own gpu codegen.
    SmallVector<Value> initial_copy;
    for (auto val : func.getArguments()) {
      initial_copy.push_back(val);
    }

    mlir::ReturnOp ret = *func.getOps<mlir::ReturnOp>().begin();
    for (auto val : ret.getOperands()) {
      initial_copy.push_back(val);
    }

    auto& func_block = func.getBody().front();
    AliasAnalysis memref_alias(&func_block, initial_copy, IsAliasOp);
    memref_alias.RunOnBlock();

    SmallVector<int32_t> offsets;
    SmallVector<int32_t> ranks;

    for (unsigned i = 0; i < launchOp.getNumKernelOperands(); ++i) {
      auto val = launchOp.getKernelOperand(i);
      offsets.push_back(memref_alias.GetLeaderIndex(val));

      if (auto memref_type = val.getType().dyn_cast<MemRefType>()) {
        ranks.push_back(memref_type.getRank());
      }
    }

    func->setAttr(getByrePrefix() + "arg_offsets", opBuilder.getI32ArrayAttr(offsets));
    func->setAttr(getByrePrefix() + "arg_ranks", opBuilder.getI32ArrayAttr(ranks));

  }
}

// Main Pass
struct GenPTXConfigPass : public GenPTXConfigBase<GenPTXConfigPass> {

  GenPTXConfigPass() : GenPTXConfigBase() {}

  void runOnFunction() override {
    auto func = getFunction();
    AddFuncAttrs(func);
  }
};

} // namespace

std::unique_ptr<FunctionPass> mlir::createGenPTXConfigPass() {
  return std::make_unique<GenPTXConfigPass>();
}
