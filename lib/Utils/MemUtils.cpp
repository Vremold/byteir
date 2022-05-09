//===- MemUtils.cpp -------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace llvm;
using namespace mlir;

Attribute mlir::wrapIntegerMemorySpace(unsigned space, MLIRContext *ctx) {
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

Optional<Value> mlir::getDimSize(OpBuilder &b, Value val, unsigned idx) {
  if (auto shapedType = val.getType().dyn_cast<ShapedType>()) {
    auto loc = val.getLoc();
    if (shapedType.isDynamicDim(idx)) {
      auto dimOp = b.create<memref::DimOp>(loc, val, idx);
      return dimOp.getResult();
    } else {
      auto cOp =
          b.create<arith::ConstantIndexOp>(loc, shapedType.getDimSize(idx));
      return cOp.getResult();
    }
  }
  return llvm::None;
}

// Create an alloc based on an existing Value 'val', with a given space.
// Return None, if not applicable.
Optional<Value> mlir::createAlloc(OpBuilder &b, Value val, unsigned space) {
  // early termination if not a memref
  if (!val.getType().isa<MemRefType>())
    return llvm::None;

  auto oldMemRefType = val.getType().cast<MemRefType>();

  auto spaceAttr = wrapIntegerMemorySpace(space, b.getContext());

  SmallVector<Value, 4> dynValue;

  auto shape = oldMemRefType.getShape();

  auto newMemRefType = MemRefType::get(shape, oldMemRefType.getElementType(),
                                       nullptr /*layout*/, spaceAttr);

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

// Get byte shift from the original allocation operation or function argument.
// Note that `shift` is different from `offset`, since `shift` is used for
// contiguous memory, while `offset` is used in multi-dimenstional situation.
// Return None, if val is not of type MemRefType or it could not be determined.
llvm::Optional<int64_t> mlir::getByteShiftFromAllocOrArgument(Value val) {
  auto memRefType = val.getType().dyn_cast_or_null<MemRefType>();
  if (!memRefType)
    return None;
  Operation *op = val.getDefiningOp();
  if (!op || isa<memref::AllocOp>(op)) {
    return 0;
  } else if (auto viewOp = dyn_cast<memref::ViewOp>(op)) {
    Value offsetVal = viewOp.byte_shift();
    if (auto offsetOp = offsetVal.getDefiningOp<arith::ConstantOp>()) {
      if (auto offsetLit =
              offsetOp.getValue().dyn_cast_or_null<IntegerAttr>()) {
        int64_t curOffset = offsetLit.getInt();
        llvm::Optional<int64_t> subOffset =
            getByteShiftFromAllocOrArgument(viewOp.source());
        if (!subOffset.hasValue())
          // the byte shift of viewOp's source is None
          return None;
        else
          return curOffset + subOffset.getValue();
      } else {
        llvm_unreachable(
            "view op's byte shift is arith.constant but not of Integer type.");
      }
    } else {
      // the byte shift of view op is not arith.constant
      return None;
    }
  } else if (auto subViewOp = dyn_cast<memref::SubViewOp>(op)) {
    return getByteShiftFromAllocOrArgument(subViewOp.source());
  } else if (auto viewLike = dyn_cast<ViewLikeOpInterface>(op)) {
    return getByteShiftFromAllocOrArgument(viewLike.getViewSource());
  }
  return None;
}

bool mlir::isStatic(MemRefType t) {
  ShapedType shape = t.dyn_cast_or_null<ShapedType>();
  if (!shape)
    return false;
  if (!shape.hasStaticShape())
    return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(t, strides, offset)))
    return false;
  if (offset == ShapedType::kDynamicStrideOrOffset)
    return false;
  for (auto stride : strides) {
    if (stride == ShapedType::kDynamicStrideOrOffset)
      return false;
  }
  return true;
}

Optional<int64_t> mlir::getSizeInBits(MemRefType t) {
  if (!isStatic(t))
    return None;
  SmallVector<int64_t> strides;
  int64_t offset;
  assert(succeeded(getStridesAndOffset(t, strides, offset)));
  int64_t numElems = offset;
  ArrayRef<int64_t> shapes = t.cast<ShapedType>().getShape();
  for (auto strideAndShape : zip(strides, shapes)) {
    int64_t stride = std::get<0>(strideAndShape);
    int64_t shape = std::get<1>(strideAndShape);
    numElems = std::max(numElems, offset + stride * shape);
  }

  auto elementType = t.getElementType();
  if (elementType.isIntOrFloat())
    return elementType.getIntOrFloatBitWidth() * numElems;

  if (auto complexType = elementType.dyn_cast<ComplexType>()) {
    elementType = complexType.getElementType();
    return elementType.getIntOrFloatBitWidth() * numElems * 2;
  }
  return None;
}

MemRefType mlir::cloneMemRefTypeWithMemSpace(MemRefType t, Attribute space) {
  return MemRefType::get(t.getShape(), t.getElementType(), t.getLayout(),
                         space);
}

MemRefType mlir::cloneMemRefTypeAndRemoveMemSpace(MemRefType t) {
  return MemRefType::get(t.getShape(), t.getElementType(), t.getLayout());
}
