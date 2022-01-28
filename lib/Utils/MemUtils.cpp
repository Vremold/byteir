//===- MemUtils.cpp ----------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"



using namespace llvm;
using namespace mlir;


Attribute mlir::wrapIntegerMemorySpace(unsigned space, MLIRContext* ctx) {
  if (space == 0)
    return nullptr;
  return IntegerAttr::get(IntegerType::get(ctx, 64), space);
}


Optional<int64_t> mlir::getRank(Value val) {
  if (auto shapedType = val.getType().dyn_cast<ShapedType>()) {
    return shapedType.getRank();
  }
  return llvm::None;
}

Optional<Value> mlir::getDimSize(OpBuilder& b, Value val, unsigned idx) {
  if (auto shapedType = val.getType().dyn_cast<ShapedType>()) {
    auto loc = val.getLoc();
    if (shapedType.isDynamicDim(idx)) {
      auto dimOp = b.create<memref::DimOp>(loc, val, idx);
      return dimOp.getResult();
    } else {
      auto cOp = b.create<arith::ConstantIndexOp>(loc, shapedType.getDimSize(idx));
      return cOp.getResult();
    }
  }
  return llvm::None;
}

// Create an alloc based on an existing Value 'val', with a given space.
// Return None, if not applicable.
Optional<Value> mlir::createAlloc(OpBuilder& b, Value val, unsigned space) {
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

