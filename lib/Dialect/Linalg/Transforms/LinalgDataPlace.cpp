//===- LinalgDataPlace.cpp ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Transforms/LinalgDataPlace.h"
#include "PassDetail.h"
#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-data-place"

namespace {

// Local utils
// Return memory space from 'memSpaces' (a list of memory space) for a gvien idx. 
// If out-of-bound, use the last value.

static int64_t getSpace(ArrayRef<int64_t> memSpaces, unsigned idx) {
  if (memSpaces.size() == 0) return getUnplacedSpace();

  if (idx < memSpaces.size()) {
    return memSpaces[idx];
  }
  return memSpaces.back();
}

// Create an alloc based on an existing Value 'val', with a given space.
// Return None, if not applicable.
static Optional<Value> createAlloc(OpBuilder& b, Value val, unsigned space) {
  // early termination if not a memref
  if (!val.getType().isa<MemRefType>()) return llvm::None;

  auto oldMemRefType = val.getType().cast<MemRefType>();

  auto spaceAttr = wrapIntegerMemorySpace(space, b.getContext());

  SmallVector<Value, 4> dynValue;

  auto shape = oldMemRefType.getShape();

  auto newMemRefType = 
    MemRefType::get(shape,
      oldMemRefType.getElementType(), nullptr/*layout*/, spaceAttr);

  for (unsigned idx = 0, n = shape.size(); idx < n; ++idx) {
    if (shape[idx] == ShapedType::kDynamicSize) {
      auto maybeValue = getDimSize(b, val, idx);
      if (!maybeValue.hasValue()) {
        return llvm::None;
      }

      dynValue.push_back(maybeValue.getValue());
    }
  }

  auto loc = val.getLoc();
  auto alloc = b.create<memref::AllocOp>(loc, newMemRefType, dynValue);
  return alloc.getResult();
}

static void dataPlaceImpl(
  OpBuilder& b, LinalgOp op) {
  if (op == nullptr) return;

  SmallVector<int64_t> memSpaces;

  if (auto arrayAttr = op->getAttrOfType<ArrayAttr>(getDataPlaceAttrName())) {
    for (auto attr : arrayAttr) {
      if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
        memSpaces.push_back(intAttr.getInt());
      } else {
        memSpaces.push_back(getUnplacedSpace());
      }
    }
  }

  auto loc = op.getLoc();
  SmallVector<Value, 4> operands;
  int idx = 0;

  // handle inputs
  for (auto input : op.inputs()) {
    int64_t space = getSpace(memSpaces, idx++);

    if (space == getUnplacedSpace()) {
      operands.push_back(input);
    } else {
      b.setInsertionPoint(op);
      auto maybeNewInput = createAlloc(b, input, space);

      if (maybeNewInput.hasValue()) {
        operands.push_back(maybeNewInput.getValue());
        // create copy
        b.create<linalg::CopyOp>(loc, input, maybeNewInput.getValue());
      } else {
        operands.push_back(input);
      }
    }
  }

  // handle outputs
  SmallVector<bool, 4> outputReplaced;
  for (auto output : op.outputs()) {
    int64_t space = getSpace(memSpaces, idx++);

    if (space == getUnplacedSpace()) {
      operands.push_back(output);
      outputReplaced.push_back(false);
    } else {
      b.setInsertionPoint(op);
      auto maybeNewInput = createAlloc(b, output, space);

      if (maybeNewInput.hasValue()) {
        operands.push_back(maybeNewInput.getValue());
        outputReplaced.push_back(true);
        // TODO check outputs as inout??
        // if so, do copy
      } else {
        operands.push_back(output);
        outputReplaced.push_back(false);
      }
    }
  }

  b.setInsertionPointAfter(op);
  auto cloned = op.clone(b, op.getLoc(), op->getResultTypes(), operands);
  cloned->removeAttr(getDataPlaceAttrName());

  idx = 0;
  int64_t numInputs = op.getNumInputs();
  for (auto output : op.outputs()) {
    if (outputReplaced[idx]) {
      // copy output
      b.create<linalg::CopyOp>(loc, operands[numInputs + idx], output);
      ++idx;
    }
  }
  
  op.erase();
}

static void collectAnchorOp(
  FuncOp func, SmallVectorImpl<LinalgOp>& collection, ArrayRef<int64_t> spaces) {
  auto ctx = func.getContext();

  // collect op with getDataPlaceAttrName as intial values
  func.walk([&](LinalgOp op) {
    // skip non-targeting or visited block
    if (op->hasAttr(getDataPlaceAttrName())) {
      
      // rewrite attribute to 'spaces' if it is UnitAttr
      if (op->hasAttrOfType<UnitAttr>(getDataPlaceAttrName())) {
        SmallVector<Attribute> arrayAttr;
        
        for (auto s : spaces) {
          arrayAttr.push_back(IntegerAttr::get(IntegerType::get(ctx, 32), s));
        }

        op->setAttr(getDataPlaceAttrName(), ArrayAttr::get(ctx, arrayAttr));
      } else if (!op->hasAttrOfType<ArrayAttr>(getDataPlaceAttrName())) {
        return;
      }
      collection.emplace_back(op);
    }
  });
}


struct LinalgDataPlacePass : public LinalgDataPlaceBase<LinalgDataPlacePass> {
  LinalgDataPlacePass() = default;
  LinalgDataPlacePass(ArrayRef<int64_t> spaces) {
    this->memSpaces = spaces;
  }

  void runOnOperation() override {
    FuncOp funcOp = getOperation();

    SmallVector<LinalgOp> collection;
    collectAnchorOp(funcOp, collection, memSpaces);

    OpBuilder b(funcOp.getContext());

    for (auto op : collection) {
      dataPlaceImpl(b, op);
    }
  }

};

} // anonymous 

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgDataPlacePass(ArrayRef<int64_t> spaces) {
  return std::make_unique<LinalgDataPlacePass>(spaces);
}
