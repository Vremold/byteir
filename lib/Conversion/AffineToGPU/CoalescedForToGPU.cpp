//===- CoalescedForToGPU.cpp ----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/AffineToGPU/AffineToGPU.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "../PassDetail.h"

#define DEBUG_TYPE "coadesced-for-to-gpu"

using namespace llvm;
using namespace mlir;
using namespace mlir::gpu;

// Some code from SCFTOGPU
namespace {

static LogicalResult checkACoalescedffineLoopMappable(
  AffineForOp forOp) {
  Region& limit = forOp.region();

  if (!areValuesDefinedAbove(forOp.getLowerBoundOperands(), limit) ||
    !areValuesDefinedAbove(forOp.getUpperBoundOperands(), limit)) {
    return forOp.emitError(
      "loop with bounds depending on other mapped loops "
      "are not supported");
  }
  return success();
}

struct CoalescedAffineLoopToGpuConverter {
  bool collectBound(AffineForOp forOp);

  void createLaunch(AffineForOp forOp, unsigned blockSize);

  // Range of the loop mapped to linearized blocks and threads.
  Value dim;
  // Lower bound of the loop mapped to linearized blocks and threads.
  Value lb;
  // Induction variable of the loop mapped to linearized blocks and threads.
  Value iv;
  // Step of the loop mapped to linearized blocks and threads.
  Value step;
};

// TODO: after newer LLVM change to CeilDivSIOp
static Value 
CreatCeilDivSIOp(OpBuilder& builder, mlir::Location loc, Value lhs, Value rhs) {
  Value constOne = builder.create<ConstantIndexOp>(loc, 1);
  Value bias = builder.create<SubIOp>(loc, rhs, constOne);
  Value sum = builder.create<AddIOp>(loc, lhs, bias);
  Value ret = builder.create<SignedDivIOp>(loc, sum, rhs);
  return ret;
}

static std::pair<Value, Value>
CreateGridAndBlock(Value dim, int64_t blockSize) {
  auto loc = dim.getLoc();
  OpBuilder builder(dim.getContext());
  builder.setInsertionPointAfter(dim.getDefiningOp());
  Value constBlock = builder.create<ConstantIndexOp>(loc, blockSize);
  Value grid = CreatCeilDivSIOp(builder, loc, dim, constBlock);
  return { grid, constBlock };
}

// TODO move another file
static Value 
CreateLinearizedIndex(OpBuilder& builder, mlir::Location loc, Value bId, Value bSize, Value tId) {
  Value mul = builder.create<MulIOp>(loc, bId, bSize);
  Value ret = builder.create<AddIOp>(loc, mul, tId);
  return ret;
}

// Replace the for with a GPU launch operation.
void CoalescedAffineLoopToGpuConverter::createLaunch(
  AffineForOp forOp,
  unsigned blockSize) {
  OpBuilder builder(forOp.getOperation());
  // Prepare the grid and block sizes for the launch operation.  If there is
  // no loop mapped to a specific dimension, use constant "1" as its size.
  Value constOne = builder.create<ConstantIndexOp>(forOp.getLoc(), 1);
  auto p = CreateGridAndBlock(dim, blockSize);

  Value gridSizeX = p.first;
  Value gridSizeY = constOne;
  Value gridSizeZ = constOne;
  Value blockSizeX = p.second;
  Value blockSizeY = constOne;
  Value blockSizeZ = constOne;

  // Create a launch op and move the body region of the innermost loop to the
  // launch op.
  auto launchOp = builder.create<gpu::LaunchOp>(
    forOp.getLoc(), gridSizeX, gridSizeY, gridSizeZ, blockSizeX,
    blockSizeY, blockSizeZ);

  // Remove the loop terminator (loops contain only a single block) 
  Operation* terminator = forOp.getBody()->getTerminator();
  terminator->erase();

  builder.setInsertionPointToStart(&launchOp.body().front());
  Value bIdx = launchOp.getBlockIds().x;
  Value id = CreateLinearizedIndex(builder, bIdx.getLoc(), bIdx,
    launchOp.getBlockSize().x, launchOp.getThreadIds().x);

  auto idLoc = id.getDefiningOp()->getLoc();
  Value cond = builder.create<CmpIOp>(idLoc, CmpIPredicate::slt, id, dim);
  auto ifOp = builder.create<scf::IfOp>(idLoc, cond, false);

  // copy body
  ifOp.getBody(0)->getOperations().splice(
    ifOp.getBody(0)->begin(), 
    forOp.getBody()->getOperations());

  // Remap the loop iterators to use block/thread identifiers 
  // with (gid * S) + LB.
  builder.setInsertionPointAfter(id.getDefiningOp());
  if (!isConstantIndex(step, 1)) {
    id = builder.create<MulIOp>(forOp.getLoc(), step, id);
  }
  Value ivReplacement =
    builder.create<AddIOp>(forOp.getLoc(), lb, id);
  iv.replaceAllUsesWith(ivReplacement);
  
  // Insert terminator
  builder.setInsertionPointToEnd(&launchOp.body().front());
  auto terminatorLoc = launchOp.body().front().back().getLoc();
  builder.create<gpu::TerminatorOp>(terminatorLoc, llvm::None);

  forOp.erase();
}

// Collect range, bound, step and induction variable in preparation for
// mapping a loop at "forOp" to a GPU kernel.
bool CoalescedAffineLoopToGpuConverter::collectBound(AffineForOp forOp) {
  OpBuilder builder(forOp.getOperation());
  auto loc = forOp.getLoc();
  lb = lowerAffineLowerBound(forOp, builder);
  Value upperBound = lowerAffineUpperBound(forOp, builder);

  if (!lb || !upperBound) {
    return false;
  }
  dim = builder.create<SubIOp>(loc, upperBound, lb); 
  step = builder.create<ConstantIndexOp>(loc, forOp.getStep());

  if (!isConstantIndex(step, 1)) {
    // dim/step  only support perfect loop for now
    dim = builder.create<SignedDivIOp>(loc, dim, step);
  }

  iv = forOp.getInductionVar();
  return true;
}

// Generic loop to GPU kernel conversion function.
static LogicalResult convertCoalescedAffineLoopToGPULaunch(AffineForOp forOp,
  unsigned blockSize) {
  if (failed(checkACoalescedffineLoopMappable(forOp))) {
    return failure();
  }

  CoalescedAffineLoopToGpuConverter converter;
  auto found_bound = converter.collectBound(forOp);
  if (!found_bound) return failure();
  converter.createLaunch(forOp, blockSize);

  return success();
}


struct CoalescedForToGPULaunchPass : public CoalescedForToGPULaunchBase<CoalescedForToGPULaunchPass> {
  CoalescedForToGPULaunchPass(int64_t bSize)
    : CoalescedForToGPULaunchBase() {
    blockSize = bSize;
  }

  void runOnFunction() final {
    auto f = getFunction();

    for (Operation& op : llvm::make_early_inc_range(f.getOps())) {
      if (auto forOp = dyn_cast<AffineForOp>(&op)) {
        if (failed(convertCoalescedAffineLoopToGPULaunch(forOp, blockSize))) {
          signalPassFailure();
        }
      }
    }
  }
};

} // namespace anonymous

std::unique_ptr<FunctionPass> mlir::createCoalescedForToGPULaunchPass(int64_t bSize) {
  return std::make_unique<CoalescedForToGPULaunchPass>(bSize);
}
