//===- CanonicalExt.cpp ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/CanonicalExt.h"
#include "byteir/Utils/AttrUtils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

template <typename ValType>
Attribute createBroadcastedDenseElementAttrImpl(
    DenseElementsAttr originAttr, ArrayRef<int64_t> originShape,
    ShapedType newType, ArrayRef<int64_t> broadcastDims) {
  SmallVector<ValType> originValues{originAttr.getValues<ValType>().begin(),
                                    originAttr.getValues<ValType>().end()};
  SmallVector<ValType> newValues;
  newValues.reserve(newType.getNumElements());
  ArrayRef<int64_t> outShape = newType.getShape();

  auto getStrides = [](ArrayRef<int64_t> shape) {
    SmallVector<int64_t> strides(shape.size(), 1);
    for (int64_t i = strides.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  };
  SmallVector<int64_t> originStrides = getStrides(originShape);
  SmallVector<int64_t> outStrides = getStrides(outShape);
  SmallVector<int64_t> dimMapping(outShape.size(), -1);
  for (size_t i = 0; i < broadcastDims.size(); ++i) {
    dimMapping[broadcastDims[i]] = i;
  }

  // return the original index and increment current index by 1.
  auto indexIncrement = [&](SmallVector<int64_t> &curIndex) {
    int64_t originIndex = 0;
    for (size_t i = 0; i < curIndex.size(); ++i) {
      if (dimMapping[i] >= 0) {
        originIndex += originStrides[dimMapping[i]] * curIndex[i];
      }
    }

    for (int64_t i = curIndex.size() - 1; i >= 0; --i) {
      curIndex[i] = (curIndex[i] + 1) % outShape[i];
      if (curIndex[i] != 0)
        break;
    }

    return originIndex;
  };

  SmallVector<int64_t> curIndex(outShape.size(), 0);
  for (int64_t i = 0; i < newType.getNumElements(); ++i) {
    int64_t originIndex = indexIncrement(curIndex);
    newValues.push_back(originValues[originIndex]);
  }
  return DenseElementsAttr::get(newType, newValues);
}

Optional<Attribute> createBroadcastedDenseElementAttr(
    DenseElementsAttr originAttr, ArrayRef<int64_t> originShape,
    ShapedType newType, ArrayRef<int64_t> broadcastDims) {
  Type elemType = originAttr.getElementType();
  if (elemType.isa<FloatType>()) {
    return createBroadcastedDenseElementAttrImpl<APFloat>(
        originAttr, originShape, newType, broadcastDims);
  } else if (elemType.isa<IntegerType>()) {
    return createBroadcastedDenseElementAttrImpl<APInt>(originAttr, originShape,
                                                        newType, broadcastDims);
  }
  return None;
}
} // namespace

LogicalResult
mlir::mhlo::EliminateSplatConstantTranspose(mhlo::TransposeOp op,
                                            PatternRewriter &rewriter) {

  if (!op.getType().hasStaticShape()) {
    return failure();
  }

  auto const_op = op.operand().getDefiningOp<mhlo::ConstOp>();
  if (!const_op) {
    return failure();
  }

  auto maybe_new_attr =
      reshapeSplatElementsAttr(const_op.value(), op.getType());
  if (!maybe_new_attr.hasValue())
    return failure();

  rewriter.replaceOpWithNewOp<mhlo::ConstOp>(op, maybe_new_attr.getValue());
  return success();
}

LogicalResult mlir::mhlo::FoldBroadcastInDim(BroadcastInDimOp op,
                                             PatternRewriter &rewriter) {
  if (!op->getResult(0).hasOneUse())
    return failure();

  Operation *broadUser = *op->getResult(0).user_begin();
  // These op types have const folding implementation,
  // in file: mlir-hlo/lib/Dialect/mhlo/IR/hlo_ops.cc
  if (!isa<AddOp, DivOp, MaxOp, MinOp, MulOp, SubOp, RemOp>(broadUser))
    return failure();

  unsigned broadOperandNumber =
      op->getResult(0).use_begin()->getOperandNumber();

  for (unsigned i = 0; i < broadUser->getNumOperands(); ++i) {
    if (i == broadOperandNumber)
      continue;
    Operation *otherOp = broadUser->getOperand(i).getDefiningOp();
    /// const_0
    ///   \
    ///   broadcast_in_dim  const_1
    ///       \            /     \ 
    ///            mul          other ops
    ///
    /// Don't fold broadcast_in_dim if const_1 has other users
    if (!otherOp || !isa<ConstOp>(otherOp) ||
        !otherOp->getResult(0).hasOneUse())
      return failure();
  }

  auto broadConstOp =
      llvm::dyn_cast_or_null<ConstOp>(op.operand().getDefiningOp());
  if (!broadConstOp)
    return failure();
  auto originAttr = broadConstOp.value().dyn_cast<DenseElementsAttr>();
  if (!originAttr)
    return failure();
  ShapedType inpType = broadConstOp.output().getType().cast<ShapedType>();
  ShapedType outputType = op->getResult(0).getType().cast<ShapedType>();
  if (!inpType.hasStaticShape() || !outputType.hasStaticShape())
    return failure();

  SmallVector<int64_t> broadcastDims;
  for (auto v : op.broadcast_dimensions()) {
    broadcastDims.push_back(v.getSExtValue());
  }
  auto newAttr = createBroadcastedDenseElementAttr(
      originAttr, inpType.getShape(), outputType, broadcastDims);
  if (!newAttr.hasValue())
    return failure();

  rewriter.replaceOpWithNewOp<ConstOp>(op, newAttr.getValue());
  return success();
}

void mlir::mhlo::getCanonicalizationExtPatterns(RewritePatternSet &results,
                                                MLIRContext *ctx) {

  // add dialect level getCanonicalizationPatterns
  auto mhlo_dailect = ctx->getOrLoadDialect<mhlo::MhloDialect>();
  if (mhlo_dailect) {
    mhlo_dailect->getCanonicalizationPatterns(results);
  }

  // add op level  getCanonicalizationPatterns
  for (RegisteredOperationName op : ctx->getRegisteredOperations()) {
    // only add mhlo-related
    if (isa<MhloDialect>(op.getDialect())) {
      op.getCanonicalizationPatterns(results, ctx);
    }
  }

  // add our extension
  results.add(mlir::mhlo::EliminateSplatConstantTranspose);
  results.add(mlir::mhlo::FoldBroadcastInDim);
}
