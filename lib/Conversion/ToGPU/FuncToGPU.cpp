//===- FuncToGPU.cpp ---------------------------------------------- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "byteir/Conversion/ToGPU/ToGPU.h"
#include "byteir/Conversion/ToGPU/Utils.h"
#include "byteir/Utils/LoopUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"
#include <utility> // pair

#define DEBUG_TYPE "func-to-gpu"

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::gpu;

namespace {

static void creaetGuardedSIMT(OpBuilder &b, Value id, Value bound,
                              LoopLikeOpInterface looplike, bool coarsen) {
  b.setInsertionPoint(looplike);

  if (coarsen) {
    addLoopLowerBound(b, looplike, id);
    multiplyLoopStep(b, looplike, bound);

    // remove attrs
    looplike->removeAttr(getLoopToSIMTAttrName());
    looplike->removeAttr(getCoarsenSIMTAttrName());

    return;
  }

  BlockAndValueMapping bvm;
  // newIV = lb + idx * step
  auto newIV = createIndexValue(b, looplike, id);
  auto oldIV = getInductionVar(looplike);
  bvm.map(oldIV, newIV);

  auto guardedBlock = createGuardedBranch(b, newIV, looplike);
  if (guardedBlock == nullptr)
    return;

  b.setInsertionPointToStart(guardedBlock);
  for (auto &block : looplike.getLoopBody()) {
    for (auto &op : block.without_terminator()) {
      b.clone(op, bvm);
    }
  }
  looplike.erase();
}

static void creaetGuardedSIMT(OpBuilder &b, FuncOp func,
                              LoopLikeOpInterface looplike,
                              GPUIndexType indexType, gpu::Dimension dim,
                              bool coarsen) {

  auto loc = looplike.getLoc();
  b.setInsertionPointToStart(&func.getBody().front());
  Value idx;
  Value bound;
  if (indexType == GPUIndexType::linear_id) {
    // idx = thread_id + block_id * block_dim
    auto bix = b.create<gpu::BlockIdOp>(loc, dim);
    auto bdim = b.create<gpu::BlockDimOp>(loc, dim);
    auto tix = b.create<gpu::ThreadIdOp>(loc, dim);
    idx = createLinearIndexValue(b, tix, bix, bdim);
    auto gdim = b.create<gpu::GridDimOp>(loc, dim);
    bound = b.create<arith::MulIOp>(loc, bdim, gdim);
  } else if (indexType == GPUIndexType::thread_id) {
    // idx = thread_id
    idx = b.create<gpu::ThreadIdOp>(loc, dim);
    bound = b.create<gpu::BlockDimOp>(loc, dim);
  } else if (indexType == GPUIndexType::block_id) {
    // idx = block_id
    idx = b.create<gpu::BlockIdOp>(loc, dim);
    bound = b.create<gpu::GridDimOp>(loc, dim);
  }

  creaetGuardedSIMT(b, idx, bound, looplike, coarsen);
}

static void convertLoopToSIMT(OpBuilder &b, FuncOp func,
                              LoopLikeOpInterface looplike) {
  auto strAttr = looplike->getAttrOfType<StringAttr>(getLoopToSIMTAttrName());
  auto coarsen = looplike->hasAttrOfType<UnitAttr>(getCoarsenSIMTAttrName());

  // default values
  GPUIndexType gpuIdxT = GPUIndexType::linear_id;
  gpu::Dimension dim = gpu::Dimension::x;

  // insert righ before loop
  if (strAttr.getValue() == getLinearIdXName()) {
    gpuIdxT = GPUIndexType::linear_id;
    dim = gpu::Dimension::x;
  } else if (strAttr.getValue() == getLinearIdYName()) {
    gpuIdxT = GPUIndexType::linear_id;
    dim = gpu::Dimension::y;
  } else if (strAttr.getValue() == getLinearIdZName()) {
    gpuIdxT = GPUIndexType::linear_id;
    dim = gpu::Dimension::z;
  } else if (strAttr.getValue() == getThreadIdXName()) {
    gpuIdxT = GPUIndexType::thread_id;
    dim = gpu::Dimension::x;
  } else if (strAttr.getValue() == getThreadIdYName()) {
    gpuIdxT = GPUIndexType::thread_id;
    dim = gpu::Dimension::y;
  } else if (strAttr.getValue() == getThreadIdZName()) {
    gpuIdxT = GPUIndexType::thread_id;
    dim = gpu::Dimension::z;
  } else if (strAttr.getValue() == getBlockIdXName()) {
    gpuIdxT = GPUIndexType::block_id;
    dim = gpu::Dimension::x;
  } else if (strAttr.getValue() == getBlockIdYName()) {
    gpuIdxT = GPUIndexType::block_id;
    dim = gpu::Dimension::y;
  } else if (strAttr.getValue() == getBlockIdZName()) {
    gpuIdxT = GPUIndexType::block_id;
    dim = gpu::Dimension::z;
  }

  creaetGuardedSIMT(b, func, looplike, gpuIdxT, dim, coarsen);
}

static void rewriteFuncImpl(OpBuilder &builder, FuncOp func) {
  SmallVector<LoopLikeOpInterface> loops;

  // collect loops from inner to outer
  func.walk([&](LoopLikeOpInterface loopLike) {
    if (loopLike->hasAttrOfType<StringAttr>(getLoopToSIMTAttrName())) {
      loops.push_back(loopLike);
    }
  });

  for (auto loop : loops) {
    convertLoopToSIMT(builder, func, loop);
  }
}

static std::pair<KernelDim3, KernelDim3> createBlockAndGrid(OpBuilder &b,
                                                            FuncOp func) {

  auto arrayAttr = func->getAttrOfType<ArrayAttr>(getToGPUAttrName());
  auto loc = func.getLoc();

  auto bx = b.create<arith::ConstantIndexOp>(
      loc, arrayAttr[0].cast<IntegerAttr>().getInt());
  auto by = b.create<arith::ConstantIndexOp>(
      loc, arrayAttr[1].cast<IntegerAttr>().getInt());
  auto bz = b.create<arith::ConstantIndexOp>(
      loc, arrayAttr[2].cast<IntegerAttr>().getInt());
  auto gx = b.create<arith::ConstantIndexOp>(
      loc, arrayAttr[3].cast<IntegerAttr>().getInt());
  auto gy = b.create<arith::ConstantIndexOp>(
      loc, arrayAttr[4].cast<IntegerAttr>().getInt());
  auto gz = b.create<arith::ConstantIndexOp>(
      loc, arrayAttr[5].cast<IntegerAttr>().getInt());

  KernelDim3 block{bx, by, bz};
  KernelDim3 grid{gx, gy, gz};

  return {block, grid};
}

static void rewriteToGPULaunchFuncImpl(OpBuilder &builder, FuncOp func,
                                       gpu::GPUFuncOp gpuFunc) {

  // erase current body
  func.getBody().front().erase();

  // create a new one body
  auto funcEntryBlock = func.addEntryBlock();

  builder.setInsertionPointToStart(funcEntryBlock);
  auto blockAndGrid = createBlockAndGrid(builder, func);
  builder.create<gpu::LaunchFuncOp>(func.getLoc(), gpuFunc, blockAndGrid.second,
                                    blockAndGrid.first, nullptr,
                                    func.getArguments());

  builder.create<mlir::ReturnOp>(func.getLoc());
}

int64_t estimateGridSize(LoopLikeOpInterface loopLike, int64_t currGs,
                         int64_t stepMultiplier) {

  auto maybeTripCnt = getConstantTripCount(loopLike, stepMultiplier);

  if (maybeTripCnt.hasValue() &&
      (maybeTripCnt.getValue() > static_cast<uint64_t>(currGs))) {
    return maybeTripCnt.getValue();
  }
  return currGs;
}

void setValidStaticGPUConfigAttr(FuncOp func, ArrayRef<int64_t> bs,
                                 ArrayRef<int64_t> gs) {

  // handle block and grid sizes
  SmallVector<int64_t> toGPUSizes;

  // read attrs
  if (auto arrayAttr = func->getAttrOfType<ArrayAttr>(getToGPUAttrName())) {
    for (auto attr : arrayAttr) {
      if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
        toGPUSizes.push_back(intAttr.getInt());
      } else {
        toGPUSizes.push_back(1);
      }
    }
  } else {
    toGPUSizes.insert(toGPUSizes.end(), bs.begin(), bs.end());
    toGPUSizes.insert(toGPUSizes.end(), gs.begin(), gs.end());
  }

  SmallVector<Attribute> toGPUAttrs;
  auto ctx = func.getContext();

  for (size_t i = 0; i < 3; ++i) {
    if (i >= toGPUSizes.size()) {
      toGPUSizes.push_back(1);
    }

    if (toGPUSizes[i] <= 0) {
      toGPUSizes[i] = 1;
    }

    auto attr = IntegerAttr::get(IntegerType::get(ctx, 32), toGPUSizes[i]);
    toGPUAttrs.push_back(attr);
  }

  // estimate maxGridSizes if possible
  SmallVector<int64_t> maxGridSizes = {0, 0, 0};
  // collect loops from inner to outer
  func.walk([&](LoopLikeOpInterface loopLike) {
    if (loopLike->hasAttrOfType<StringAttr>(getLoopToSIMTAttrName())) {
      auto coarsen =
          loopLike->hasAttrOfType<UnitAttr>(getCoarsenSIMTAttrName());
      if (coarsen)
        return;

      auto strAttr =
          loopLike->getAttrOfType<StringAttr>(getLoopToSIMTAttrName());

      if (strAttr.getValue() == getLinearIdXName()) {
        maxGridSizes[0] =
            estimateGridSize(loopLike, maxGridSizes[0], toGPUSizes[0]);
      } else if (strAttr.getValue() == getLinearIdYName()) {
        maxGridSizes[1] =
            estimateGridSize(loopLike, maxGridSizes[1], toGPUSizes[1]);
      } else if (strAttr.getValue() == getLinearIdZName()) {
        maxGridSizes[2] =
            estimateGridSize(loopLike, maxGridSizes[2], toGPUSizes[2]);
      } else if (strAttr.getValue() == getBlockIdXName()) {
        maxGridSizes[0] = estimateGridSize(loopLike, maxGridSizes[0], 1);
      } else if (strAttr.getValue() == getBlockIdYName()) {
        maxGridSizes[1] = estimateGridSize(loopLike, maxGridSizes[1], 1);
      } else if (strAttr.getValue() == getBlockIdZName()) {
        maxGridSizes[2] = estimateGridSize(loopLike, maxGridSizes[2], 1);
      }
    }
  });

  for (size_t i = 0; i < 3; ++i) {
    size_t j = i + 3;
    // if no estimation use suggested attr value
    if (maxGridSizes[i] == 0) {
      if (j < toGPUSizes.size() && toGPUSizes[j] > 0) {
        maxGridSizes[i] = toGPUSizes[j];
      } else {
        maxGridSizes[i] = 1;
      }
    }

    auto attr = IntegerAttr::get(IntegerType::get(ctx, 32), maxGridSizes[i]);
    toGPUAttrs.push_back(attr);
  }

  func->setAttr(getToGPUAttrName(), ArrayAttr::get(ctx, toGPUAttrs));
}

struct ConvertFuncToGPUPass
    : public ConvertFuncToGPUBase<ConvertFuncToGPUPass> {
  ConvertFuncToGPUPass(ArrayRef<int64_t> bs, ArrayRef<int64_t> gs,
                       const std::string &name)
      : ConvertFuncToGPUBase() {
    this->blockSizes = bs;
    this->gridSizes = gs;
    this->moduleName = name;
  }

  void runOnOperation() final {

    // early termination if no anchor or no moduleName
    if (moduleName.empty()) {
      return;
    }

    ModuleOp m = getOperation();
    SmallVector<FuncOp> funcCollector;

    // collect all anchored function
    for (auto func : m.getOps<FuncOp>()) {
      if (func->hasAttr(getToGPUAttrName())) {
        setValidStaticGPUConfigAttr(func, blockSizes, gridSizes);
        funcCollector.push_back(func);
      }
    }

    // early termination if no anchored func
    if (funcCollector.empty()) {
      return;
    }

    auto gm = getOrCreateGPUModule(m, moduleName);
    SymbolTable gmTable(gm);

    OpBuilder builder(gm.getContext());

    // create GPUFuncOp and gpu::LaunchFunc
    for (auto func : funcCollector) {
      rewriteFuncImpl(builder, func);
      auto gpuFunc = cloneFuncToGPUFunc(builder, func, gm);
      gmTable.insert(gpuFunc);
      rewriteToGPULaunchFuncImpl(builder, func, gpuFunc);

      // remove attr
      func->removeAttr(getToGPUAttrName());
    }

    // set gpu.container_module
    m->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
               UnitAttr::get(m.getContext()));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertFuncToGPUPass(ArrayRef<int64_t> bs, ArrayRef<int64_t> gs,
                                 const std::string &name) {

  return std::make_unique<ConvertFuncToGPUPass>(bs, gs, name);
}
