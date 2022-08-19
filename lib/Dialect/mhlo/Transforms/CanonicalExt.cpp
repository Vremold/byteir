//===- CanonicalExt.cpp ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/CanonicalExt.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>

#define DEBUG_TYPE "canonical-ext"

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

// FIXME: this pattern should move to shape dialect
LogicalResult mlir::mhlo::foldShapeBroadcast(shape::BroadcastOp op,
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

LogicalResult mlir::mhlo::foldBroadcastInDim(BroadcastInDimOp op,
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

namespace {

struct ConcatChunk {
  bool isSlice;  // specify whether from slice or not
  int64_t axis;  // concat axis
  int64_t begin; // concat begin along the concat axis
  int64_t end;   // concat end along the concat axis
  Value val; // source val, either slice source if from slice, or concat source
             // if not from slice
  SmallVector<unsigned> ids; // concat's arg id

  ConcatChunk(Value v, int64_t id)
      : isSlice(false), axis(-1), begin(-1), end(-1), val(v) {
    ids.push_back(id);
  }

  ConcatChunk(Value v, int64_t a, int64_t b, int64_t e, int64_t id)
      : isSlice(true), axis(a), begin(b), end(e), val(v) {
    ids.push_back(id);
  }
};

static ConcatChunk getChunkOfSlice(unsigned id, mhlo::ConcatenateOp concat,
                                   mhlo::SliceOp slice) {

  uint64_t dim = concat.dimension();
  const auto &concatShape = concat.getType().getShape();
  const auto &sliceShape = slice.getType().getShape();

  auto val = slice.getOperand();

  if (auto valTy = val.getType().dyn_cast<TensorType>()) {
    const auto &valShape = valTy.getShape();

    if (concatShape.size() == sliceShape.size() &&
        sliceShape.size() == valShape.size()) {
      // only support equal rank

      bool isSupport = true;
      int64_t begin = -1;
      int64_t end = -1;

      auto startAttr = slice.start_indices();
      auto limitAttr = slice.limit_indices();
      auto stridesAttr = slice.strides();

      for (unsigned i = 0; i < concatShape.size(); ++i) {
        const int64_t start = startAttr.getValues<IntegerAttr>()[i].getInt();
        const int64_t limit = limitAttr.getValues<IntegerAttr>()[i].getInt();
        const int64_t stride = stridesAttr.getValues<IntegerAttr>()[i].getInt();

        if (i == dim) {
          if (stride == 1) {
            begin = start;
            end = limit;
          } else {
            isSupport = false;
            break;
          }
        } else {
          if (start != 0 || limit != concatShape[i] || stride != 1) {
            isSupport = false;
            break;
          }
        }
      }

      if (isSupport) {
        return ConcatChunk(val, dim, begin, end, id);
      }
    } // equal rank
  }

  return ConcatChunk(val, id);
}

static void computeBeginAndEnd(const ConcatChunk &chunk, size_t dim,
                               SmallVectorImpl<int64_t> &begins,
                               SmallVectorImpl<int64_t> &ends) {

  if (auto inputTy = chunk.val.getType().dyn_cast<TensorType>()) {
    const auto &shape = inputTy.getShape();

    for (size_t i = 0; i < shape.size(); ++i) {
      if (i == dim) {
        begins[i] = chunk.begin;
        ends[i] = chunk.end;
      } else {
        begins[i] = 0;
        ends[i] = shape[i];
      }
    }
  };
}

} // namespace

///  Fold concatenate of continuous slices
///  FIXME: support static only for now, relax it later
LogicalResult
mlir::mhlo::foldConcatWithContinuousSlices(mhlo::ConcatenateOp op,
                                           PatternRewriter &rewriter) {

  // support static now
  if (!op.getType().hasStaticShape()) {
    LLVM_DEBUG(llvm::dbgs() << "concat has no static shape\n");
    return failure();
  }
  uint64_t dim = op.dimension();

  SmallVector<ConcatChunk> chunks;
  bool hasMerged = false;
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    if (auto slice = op.getOperand(i).getDefiningOp<mhlo::SliceOp>()) {
      // handle 1D slice only along dim axis
      auto chunk = getChunkOfSlice(i, op, slice);

      if (!chunks.empty() && (chunks.back().val == chunk.val) &&
          (chunks.back().axis == chunk.axis) &&
          (chunks.back().end == chunk.begin)) {
        chunks.back().end = chunk.end;
        chunks.back().ids.push_back(i);
        hasMerged = true;
      } else {
        chunks.push_back(chunk);
      }
    } else {
      chunks.push_back(ConcatChunk(op.getOperand(i), i));
    }
  }

  if (!hasMerged) {
    LLVM_DEBUG(llvm::dbgs() << "concat has no mergable slices\n");
    return failure();
  }

  // Only handle one chunk for now
  // TODO: add support to multiple chunk
  if (chunks.size() > 1) {
    for (size_t i = 0; i < chunks.size(); ++i) {
      auto &c = chunks[i];
      LLVM_DEBUG(llvm::dbgs() << "chunk " << i << "\n");
      LLVM_DEBUG(llvm::dbgs() << "slice axis " << c.axis << "\n");
      LLVM_DEBUG(llvm::dbgs() << "slice begin " << c.begin << "\n");
      LLVM_DEBUG(llvm::dbgs() << "slice end " << c.end << "\n");
      LLVM_DEBUG(llvm::dbgs() << "operand id from " << c.ids.front() << " to "
                              << c.ids.back() << "\n");
    }
    return failure();
  }

  // only one chunk case
  // chunks.size() == 1
  auto concatTy = op.getType();
  const auto &chunk = chunks.back();
  // either identity or 1 slice
  auto extent = concatTy.getShape()[dim];
  if (auto inputTy = chunk.val.getType().dyn_cast<TensorType>()) {
    if (inputTy == concatTy && chunk.begin == 0 && chunk.end == extent) {
      // identity
      rewriter.replaceOp(op, chunk.val);
    } else {
      // 1 slice
      int64_t rank = op.getType().getRank();
      auto indicesTy = RankedTensorType::get(rank, rewriter.getI64Type());

      SmallVector<int64_t> begins(rank, 0);
      SmallVector<int64_t> ends(rank, 0);

      // FIXME: support unit-stride now
      SmallVector<int64_t> strides(rank, 1);

      computeBeginAndEnd(chunk, dim, begins, ends);

      rewriter.replaceOpWithNewOp<mhlo::SliceOp>(
          op, chunk.val, DenseIntElementsAttr::get(indicesTy, begins),
          DenseIntElementsAttr::get(indicesTy, ends),
          DenseIntElementsAttr::get(indicesTy, strides));
    }
    return success();
  }
  return failure();
}

template <typename Op, template <typename> typename Func>
LogicalResult mlir::mhlo::foldLargeBinaryOp(Op op, PatternRewriter &rewriter) {
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

// mhlo.dynamic_conv => mhlo.convolution canonicalization
LogicalResult mlir::mhlo::simplifyDynamicConvToConv(mhlo::DynamicConvOp op,
                                                    PatternRewriter &rewriter) {
  DenseIntElementsAttr paddingAttr;
  if (!matchPattern(op.d_padding(), m_Constant(&paddingAttr))) {
    return failure();
  }
  if (paddingAttr.isSplat() && paddingAttr.getSplatValue<APInt>().isZero()) {
    mhlo::ConvolutionOp convOp = rewriter.create<mhlo::ConvolutionOp>(
        op->getLoc(), op.getType(), llvm::ArrayRef<Value>{op.lhs(), op.rhs()},
        op->getAttrs());
    rewriter.replaceOp(op, convOp.getResult());
    return success();
  } else {
    assert(!op->hasAttr("padding"));
    auto padding = llvm::to_vector(
        llvm::map_range(paddingAttr.getValues<APInt>(),
                        [&](APInt i) { return i.getSExtValue(); }));
    SmallVector<NamedAttribute> attrs = llvm::to_vector(op->getAttrs());
    attrs.push_back(NamedAttribute(
        rewriter.getStringAttr("padding"),
        getI64ElementsAttr(padding,
                           {2, static_cast<int64_t>(padding.size()) / 2},
                           &rewriter)));
    mhlo::ConvolutionOp convOp = rewriter.create<mhlo::ConvolutionOp>(
        op->getLoc(), op.getType(), llvm::ArrayRef<Value>{op.lhs(), op.rhs()},
        attrs);
    rewriter.replaceOp(op, convOp.getResult());
    return success();
  }
  return failure();
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
  results.add(mlir::mhlo::foldBroadcastInDim);
  results.add(mlir::mhlo::foldConcatWithContinuousSlices);
  results.add(mlir::mhlo::foldShapeBroadcast);
  results.add(mlir::mhlo::simplifyDynamicConvToConv);
  results.add(mlir::mhlo::foldLargeBinaryOp<mhlo::AddOp, std::plus>);
  results.add(mlir::mhlo::foldLargeBinaryOp<mhlo::MulOp, std::multiplies>);
  results.add(mlir::mhlo::foldLargeBinaryOp<mhlo::SubtractOp, std::minus>);
  results.add(mlir::mhlo::foldLargeBinaryOp<mhlo::DivOp, Divide>);
  results.add(mlir::mhlo::foldLargeBinaryOp<mhlo::RemOp, Remainder>);
  results.add(mlir::mhlo::foldLargeBinaryOp<mhlo::MaxOp, Max>);
  results.add(mlir::mhlo::foldLargeBinaryOp<mhlo::MinOp, Min>);
}
