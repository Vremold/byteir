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
#include "byteir/Dialect/mhlo/Transforms/ElementFusion.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Parser.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

using namespace byteir;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::byre;
using namespace mlir::gpu;
using namespace mlir::memref;
using namespace llvm;

namespace {

bool IsAliasOp(Operation &op) {
  return isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::ReshapeOp>(
      op);
};

// support static for now
// TODO extend it to support dynamic block/grid sizes
// TODO unify CUDA/PTX into the same pass with compilation option
static void AddFuncAttrs(FuncOp func) {
  // handle elementwise fusion
  if (func->hasAttr(getByteIRElementwiseFusionAttrName())) {
    mlir::OpBuilder opBuilder(func);

    if (func.getOps<gpu::LaunchFuncOp>().empty())
      return;

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
    func->setAttr(getByreForceComputeNameAttrName(), opBuilder.getUnitAttr());

    // Handle arg mapping here
    // LWC: this is tentative when we are using GPU Kernel Outlining.
    // TODO: drop this when we are arrange our arg placement in our own gpu
    // codegen.
    SmallVector<Value> initial_copy;
    for (auto val : func.getArguments()) {
      initial_copy.push_back(val);
    }

    mlir::ReturnOp ret = *func.getOps<mlir::ReturnOp>().begin();
    for (auto val : ret.getOperands()) {
      initial_copy.push_back(val);
    }

    auto &func_block = func.getBody().front();
    AliasAnalysis memref_alias(&func_block, initial_copy, IsAliasOp);
    memref_alias.RunOnBlock();

    SmallVector<int32_t> offsets;
    SmallVector<int32_t> ranks;
    SmallDenseSet<int> visited;

    for (unsigned i = 0; i < launchOp.getNumKernelOperands(); ++i) {
      auto val = launchOp.getKernelOperand(i);
      int index = memref_alias.GetLeaderIndex(val);
      offsets.push_back(index);
      visited.insert(index);
      if (auto memref_type = val.getType().dyn_cast<MemRefType>()) {
        ranks.push_back(memref_type.getRank());
      }
    }

    // handle unused alias args
    SmallVector<int32_t> unused_alias;
    for (unsigned i = 0; i < initial_copy.size(); ++i) {
      // skip visisted
      if (visited.contains(i))
        continue;

      auto val = initial_copy[i];
      int index = memref_alias.GetLeaderIndex(val);

      unused_alias.push_back(i);
      unused_alias.push_back(index);
    }

    func->setAttr(getByreArgOffsetAttrName(),
                  opBuilder.getI32ArrayAttr(offsets));

    func->setAttr(getByrePrefix() + getByreArgRankAttrName(),
                  opBuilder.getI32ArrayAttr(ranks));

    if (!unused_alias.empty()) {
      func->setAttr(getByrePassThroughArgAttrName(),
                    opBuilder.getI32ArrayAttr(unused_alias));
    }
  }
}

// Main Pass
struct GenPTXConfigPass : public GenPTXConfigBase<GenPTXConfigPass> {

  GenPTXConfigPass() : GenPTXConfigBase() {}

  void runOnOperation() override {
    FuncOp func = getOperation();
    AddFuncAttrs(func);
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createGenPTXConfigPass() {
  return std::make_unique<GenPTXConfigPass>();
}
