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

