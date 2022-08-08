//===- CanonicalExt.cpp ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>

#include "byteir/Dialect/mhlo/Transforms/CanonicalExt.h"
#include "byteir/Utils/AttrUtils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"

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

static const APFloat &addSign(const APFloat &v, Type) { return v; }
static APSInt addSign(const APInt &v, Type t) {
  // Add signedness information to the value, treating signless as signed.
  return APSInt(v, t.isUnsignedInteger());
}

// this function copyed from mlir-hlo/lib/Dialect/mhlo/IR/hlo_ops.cc
template <typename Op, typename ElementType = Type, typename ValType,
          typename Convert>
static Attribute BinaryFolder(Op *op, ArrayRef<Attribute> attrs) {
  if (!attrs[0] || !attrs[1])
    return {};

  DenseElementsAttr lhs = attrs[0].dyn_cast<DenseElementsAttr>();
  DenseElementsAttr rhs = attrs[1].dyn_cast<DenseElementsAttr>();
  if (!lhs || !rhs)
    return {};

  ShapedType type = op->getType().template cast<ShapedType>();
  if (!type.hasStaticShape()) {
    return {};
  }

  Type etype = type.getElementType();

  // Evaluate for integer values.
  if (!etype.isa<ElementType>()) {
    return {};
  }

  // Special case for folding splats no matter how large.
  // Only covers the case of both attrs being splats; operation-specific cases
  // like adding a zero or multiplying by one are handled elsewhere.
  SplatElementsAttr splatLhs = lhs.dyn_cast<SplatElementsAttr>();
  SplatElementsAttr splatRhs = rhs.dyn_cast<SplatElementsAttr>();
  if (splatLhs && splatRhs) {
    auto signedLhs = addSign(splatLhs.getSplatValue<ValType>(), etype);
    auto signedRhs = addSign(splatRhs.getSplatValue<ValType>(), etype);
    FailureOr<decltype(signedLhs)> result(Convert()(signedLhs, signedRhs));
    return succeeded(result) ? SplatElementsAttr::get(type, *result)
                             : Attribute();
  }

  SmallVector<ValType, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<ValType>(), rhs.getValues<ValType>())) {
    auto signedLhs = addSign(std::get<0>(zip), etype);
    auto signedRhs = addSign(std::get<1>(zip), etype);
    FailureOr<decltype(signedLhs)> result(Convert()(signedLhs, signedRhs));
    if (failed(result)) {
      return {};
    }
    values.push_back(std::move(*result));
  }

  return DenseElementsAttr::get(type, values);
}

template <typename T> struct Divide : std::divides<T> {};

template <> struct Divide<APSInt> {
  FailureOr<APSInt> operator()(const APSInt &a, const APSInt &b) const {
    if (b.isZero())
      return failure();
    return a / b;
  }
};

template <typename T> struct Remainder : std::modulus<T> {};

template <> struct Remainder<APSInt> {
  FailureOr<APSInt> operator()(const APSInt &a, const APSInt &b) const {
    if (b.isZero())
      return failure();
    return a % b;
  }
};

template <> struct Remainder<APFloat> {
  APFloat operator()(const APFloat &a, const APFloat &b) const {
    APFloat result(a);
    result.remainder(b);
    return result;
  }
};

template <typename T> struct Max {
  T operator()(const T &a, const T &b) const { return std::max<T>(a, b); }
};

template <typename T> struct Min {
  T operator()(const T &a, const T &b) const { return std::min<T>(a, b); }
};

template <typename T> struct And {
  T operator()(const T &a, const T &b) const { return a & b; }
};

template <typename T> struct Or {
  T operator()(const T &a, const T &b) const { return a | b; }
};

template <typename T> struct Xor {
  T operator()(const T &a, const T &b) const { return a ^ b; }
};

} // namespace

LogicalResult
mlir::mhlo::EliminateSplatConstantTranspose(mhlo::TransposeOp op,
                                            PatternRewriter &rewriter) {

  if (!op.getType().hasStaticShape()) {
    return failure();
  }

  auto const_op = op.operand().getDefiningOp<mhlo::ConstantOp>();
  if (!const_op) {
    return failure();
  }

  auto maybe_new_attr =
      reshapeSplatElementsAttr(const_op.value(), op.getType());
  if (!maybe_new_attr.hasValue())
    return failure();

  rewriter.replaceOpWithNewOp<mhlo::ConstantOp>(op, maybe_new_attr.getValue());
  return success();
}

// FIXME: this pattern should move to shape dialect
LogicalResult mlir::mhlo::FoldShapeBroadcast(shape::BroadcastOp op,
                                             PatternRewriter &rewriter) {

  SmallVector<SmallVector<int64_t>> shapes;
  SmallVector<Value> values;
  for (auto shape : op.getShapes()) {
    values.push_back(shape);
    if (auto inputShape = shape.getDefiningOp<shape::ShapeOfOp>()) {
      if (auto shapeType =
              inputShape.getArg().getType().dyn_cast<ShapedType>()) {
        shapes.push_back(llvm::to_vector(shapeType.getShape()));
      } else {
        return failure();
      }
    } else if (auto inputShape = shape.getDefiningOp<shape::ConstShapeOp>()) {
      shapes.push_back(llvm::to_vector(
          llvm::map_range(inputShape.getShape(), [](APInt elem) {
            return static_cast<int64_t>(elem.getZExtValue());
          })));
    } else {
      return failure();
    }
  }
  // do broadcast
  // see definition in https://mlir.llvm.org/docs/Dialects/ShapeDialect/
  size_t size = 0;
  for (auto &&shape : shapes) {
    size = std::max(shape.size(), size);
  }
  auto copyShapes = shapes;
  for (auto &shape : shapes) {
    if (shape.size() < size) {
      shape.insert(shape.begin(), size - shape.size(), 1);
    }
  }
  for (size_t i = 0; i < size; ++i) {
    int64_t res = 1;
    for (auto &shape : shapes) {
      if (shape[i] > 1) {
        res = std::max(shape[i], res);
      } else if (shape[i] == ShapedType::kDynamicSize && res == 1) {
        res = ShapedType::kDynamicSize;
      }
    }
    for (auto &shape : shapes) {
      shape[i] = res;
    }
  }
  // if output shape equal to the value shape, replace with SSA value
  int index = 0;
  for (auto &&shapePair : llvm::zip(shapes, copyShapes)) {
    if (std::get<0>(shapePair).size() != std::get<1>(shapePair).size()) {
      continue;
    }
    if (llvm::all_of(llvm::zip(std::get<0>(shapePair), std::get<1>(shapePair)),
                     [](auto dimPair) {
                       return std::get<0>(dimPair) == std::get<1>(dimPair);
                     })) {
      rewriter.replaceOp(op, values[index]);
      return success();
    }
    index += 1;
  }
  return failure();
}

LogicalResult mlir::mhlo::FoldBroadcastInDim(BroadcastInDimOp op,
                                             PatternRewriter &rewriter) {
  if (!op->getResult(0).hasOneUse())
    return failure();

  Operation *broadUser = *op->getResult(0).user_begin();
  // These op types have const folding implementation,
  // in file: mlir-hlo/lib/Dialect/mhlo/IR/hlo_ops.cc
  if (!isa<AddOp, DivOp, MaxOp, MinOp, MulOp, SubtractOp, RemOp>(broadUser))
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
    if (!otherOp || !isa<ConstantOp>(otherOp) ||
        !otherOp->getResult(0).hasOneUse())
      return failure();
  }

  auto broadConstOp =
      llvm::dyn_cast_or_null<ConstantOp>(op.operand().getDefiningOp());
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

  rewriter.replaceOpWithNewOp<ConstantOp>(op, newAttr.getValue());
  return success();
}

template <typename Op, template <typename> typename Func>
LogicalResult mlir::mhlo::FoldLargeBinaryOp(Op op, PatternRewriter &rewriter) {
  auto lhsOp = op.lhs().template getDefiningOp<mhlo::ConstantOp>();
  auto rhsOp = op.rhs().template getDefiningOp<mhlo::ConstantOp>();
  if (!lhsOp || !rhsOp) {
    return failure();
  }
  RankedTensorType type = op.getType().template dyn_cast<RankedTensorType>();
  if (!type || !type.hasStaticShape()) {
    return failure();
  }

  Attribute result;
  if (type.getElementType().isa<FloatType>()) {
    result = BinaryFolder<Op, FloatType, APFloat, Func<APFloat>>(
        &op, ArrayRef<Attribute>{lhsOp.value(), rhsOp.value()});
  } else if (type.getElementType().isa<IntegerType>()) {
    result = BinaryFolder<Op, IntegerType, APInt, Func<APSInt>>(
        &op, ArrayRef<Attribute>{lhsOp.value(), rhsOp.value()});
  }
  if (!result) {
    return failure();
  }
  mhlo::ConstantOp newConstant =
      rewriter.create<mhlo::ConstantOp>(op->getLoc(), result);
  rewriter.replaceOp(op, newConstant.output());
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
  results.add(mlir::mhlo::FoldShapeBroadcast);
  results.add(mlir::mhlo::FoldLargeBinaryOp<mhlo::AddOp, std::plus>);
  results.add(mlir::mhlo::FoldLargeBinaryOp<mhlo::MulOp, std::multiplies>);
  results.add(mlir::mhlo::FoldLargeBinaryOp<mhlo::SubtractOp, std::minus>);
  results.add(mlir::mhlo::FoldLargeBinaryOp<mhlo::DivOp, Divide>);
  results.add(mlir::mhlo::FoldLargeBinaryOp<mhlo::RemOp, Remainder>);
  results.add(mlir::mhlo::FoldLargeBinaryOp<mhlo::MaxOp, Max>);
  results.add(mlir::mhlo::FoldLargeBinaryOp<mhlo::MinOp, Min>);
}
