//===- GenPTXConfig.cpp ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "byteir/Conversion/ToByre/Common.h"
#include "byteir/Conversion/ToPTX/ToPTX.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

using namespace mlir;
using namespace mlir::byre;
using namespace mlir::gpu;
using namespace llvm;

namespace {

// support static for now
// TODO extend it to support dynamic block/grid sizes
// TODO unify CUDA/PTX into the same pass with compilation option

static void AddFuncAttrs(FuncOp func) {
  // handle elementwise fusion
  if (func->hasAttr(getByreElementwiseFusionName())) {
    
    mlir::OpBuilder opBuilder(func);

    for(auto launchOp : func.getOps<gpu::LaunchFuncOp>()) {

      func->setAttr(getByrePrefix() + "kernel_name",
        opBuilder.getStringAttr(launchOp.getKernelName().getValue()));

      // Handle 1D only, since element-wise is only using 1D (linearized)
      auto grid = launchOp.getGridSizeOperandValues();
      int64_t gx = cast<ConstantIndexOp>(grid.x.getDefiningOp()).getValue();
      func->setAttr(getByrePrefix() + "GridSize.x",
        opBuilder.getIntegerAttr(opBuilder.getIntegerType(32), gx));

      auto block = launchOp.getBlockSizeOperandValues();
      int64_t bx = cast<ConstantIndexOp>(block.x.getDefiningOp()).getValue();
      func->setAttr(getByrePrefix() + "BlockSize.x",
        opBuilder.getIntegerAttr(opBuilder.getIntegerType(32), bx));
    }

    func->setAttr(getByreComputeName(), opBuilder.getStringAttr("PTXOp"));
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