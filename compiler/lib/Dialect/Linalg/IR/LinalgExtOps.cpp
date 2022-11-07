//===- LinalgExtOps.cpp ---------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//
// Some code comes from LinalgExtOps.cpp in IREE project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Some code from LinalgOps.cpp in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SMLoc.h"

#include <iostream>

using namespace mlir;
using namespace mlir::linalg_ext;

#include "byteir/Dialect/Linalg/IR/LinalgExtOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Linalg dialect.
//===----------------------------------------------------------------------===//

void mlir::linalg_ext::LinalgExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "byteir/Dialect/Linalg/IR/LinalgExtOps.cpp.inc"
      >();
}

//////////////////////////////
// local utils
//////////////////////////////
namespace {

// move to affine util
static AffineMap getMultiDimIdentityMapWithSkip(unsigned numDims, unsigned skip,
                                                MLIRContext *context) {
  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    if (i == skip)
      continue;
    dimExprs.push_back(mlir::getAffineDimExpr(i, context));
  }
  return AffineMap::get(/*dimCount=*/numDims, /*symbolCount=*/0, dimExprs,
                        context);
}

static void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, ValueRange inputBuffers, ValueRange outputBuffers) {
  for (Value value : results) {
    effects.emplace_back(MemoryEffects::Allocate::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : inputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : outputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), value,
                         SideEffects::DefaultResource::get());
  }
}

/// Returns a memref.subview or a tensor.extract_slice based on the type of the
/// `source`.
static Value getSlice(OpBuilder &b, Location loc, Value source,
                      ArrayRef<OpFoldResult> offsets,
                      ArrayRef<OpFoldResult> sizes,
                      ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Value>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                                strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        return b.create<memref::SubViewOp>(loc, source, offsets, sizes,
                                           strides);
      })
      .Default([&](Type t) { return nullptr; });
}

} // namespace

//////////////////////////////
// global utils
//////////////////////////////
Value mlir::linalg_ext::getDimValue(OpBuilder &builder, Location loc, Value v,
                                    int64_t dim) {
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return builder.create<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType t) -> Value {
        return builder.create<memref::DimOp>(loc, v, dim);
      })
      .Default([&](Type t) { return Value(); });
}

OpFoldResult mlir::linalg_ext::getDim(OpBuilder &builder, Location loc, Value v,
                                      int64_t dim) {
  auto t = v.getType().cast<ShapedType>();
  if (t.isDynamicDim(dim)) {
    return getDimValue(builder, loc, v, dim);
  }
  return builder.getI64IntegerAttr(t.getDimSize(dim));
}

//////////////////////////////
// CustomOp
//////////////////////////////

mlir::LogicalResult mlir::linalg_ext::CustomOp::verify() {
  // FIXME
  return success();
}

llvm::SmallVector<utils::IteratorType>
mlir::linalg_ext::CustomOp::getLoopIteratorTypes() {
  // FIXME
  return {};
}

ArrayAttr mlir::linalg_ext::CustomOp::getIndexingMaps() {
  // FIXME
  return ArrayAttr();
}

SmallVector<Range>
mlir::linalg_ext::CustomOp::getIterationDomain(class mlir::OpBuilder &) {
  // FIXME
  return {};
}

SmallVector<Operation *> mlir::linalg_ext::CustomOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  // a fake tiling
  // FIXME
  SmallVector<Operation *> res;
  res.push_back(this->getOperation());
  return res;
}

LogicalResult mlir::linalg_ext::CustomOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  return success();
}

//////////////////////////////
// ScanOp
//////////////////////////////

mlir::LogicalResult mlir::linalg_ext::ScanOp::verify() {
  Operation *op = getOperation();
  if (getNumInputs() != 1) {
    return op->emitOpError("expected one input operands");
  }
  if (getNumOutputs() != 2) {
    return op->emitOpError("expected two output operands");
  }
  if (!input().getType().isa<ShapedType>()) {
    return op->emitOpError("expected first input element type to be shaped");
  }
  auto accumulatorType = accumulator().getType().cast<ShapedType>();
  auto inputType = input().getType().cast<ShapedType>();
  auto outputType = output().getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShapes = inputType.getShape();
  ArrayRef<int64_t> outputShapes = outputType.getShape();
  if (accumulatorType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/accumulator element types to be identical");
  }
  ArrayRef<int64_t> accumulatorShape = accumulatorType.getShape();
  int64_t accumulatorRank = accumulatorType.getRank();
  if (accumulatorRank != inputType.getRank() - 1) {
    return op->emitOpError(
        "expected accumulator rank to be equal to input rank - 1");
  }
  SmallVector<int64_t> expectedAccumulatorShape;
  for (int i = 0; i < inputType.getRank(); i++) {
    if (i != getDimension())
      expectedAccumulatorShape.push_back(inputShapes[i]);
  }
  if (llvm::any_of(llvm::zip(expectedAccumulatorShape, accumulatorShape),
                   [](std::tuple<int64_t, int64_t> s) {
                     return std::get<0>(s) != ShapedType::kDynamicSize &&
                            std::get<1>(s) != ShapedType::kDynamicSize &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op->emitOpError("incompatible input/accumulator shapes");
  }
  if (inputType.getElementType() != outputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  if (inputShapes.size() != outputShapes.size()) {
    return op->emitOpError("expected input/output to have identical ranks");
  }
  if (llvm::any_of(llvm::zip(inputShapes, outputShapes),
                   [](std::tuple<int64_t, int64_t> s) {
                     return std::get<0>(s) != ShapedType::kDynamicSize &&
                            std::get<1>(s) != ShapedType::kDynamicSize &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op->emitOpError("incompatible input/output shapes");
  }
  return success();
}

SmallVector<utils::IteratorType>
mlir::linalg_ext::ScanOp::getLoopIteratorTypes() {
  // All loops except the dimension are parallel.
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

ArrayAttr mlir::linalg_ext::ScanOp::getIndexingMaps() {
  unsigned rank = getOperandRank();
  SmallVector<AffineMap> maps;
  MLIRContext *ctx = getContext();

  // input
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));

  // outputs
  // result
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
  unsigned dim = getDimension();
  // accum
  maps.push_back(getMultiDimIdentityMapWithSkip(rank, dim, ctx));

  return Builder(ctx).getAffineMapArrayAttr(maps);
}

SmallVector<Range>
mlir::linalg_ext::ScanOp::getIterationDomain(class mlir::OpBuilder &builder) {
  int64_t operandRank = getOperandRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = input();
  for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<Operation *>
mlir::linalg_ext::ScanOp::getTiledImplementation(OpBuilder &builder,
                                                 ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getOperandRank();
  assert(offsets.size() == static_cast<size_t>(rank) &&
         sizes.size() == static_cast<size_t>(rank));
  auto oneAttr = builder.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> strides(rank, oneAttr);
  Location loc = getLoc();
  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), input(), offsets, sizes, strides));
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), getOutputs()[0], offsets, sizes, strides));
  if (rank > 1) {
    SmallVector<OpFoldResult> accumOffsets, accumSizes;
    if (failed(getResultTilePosition(builder, 1, offsets, sizes, accumOffsets,
                                     accumSizes))) {
      return {};
    }
    SmallVector<OpFoldResult> accumStrides(rank - 1, oneAttr);
    tiledOperands.emplace_back(getSlice(builder, getLoc(), getOutputs()[1],
                                        accumOffsets, accumSizes,
                                        accumStrides));
  } else {
    tiledOperands.emplace_back(getOutputs()[1]);
  }

  SmallVector<Type, 4> resultTypes;
  if (hasTensorSemantics()) {
    resultTypes.push_back(tiledOperands[1].getType());
    resultTypes.push_back(tiledOperands[2].getType());
  }

  Operation *tiledScanOp = cast<DestinationStyleOpInterface>(getOperation())
                               .clone(builder, loc, resultTypes, tiledOperands);
  return {tiledScanOp};
}

LogicalResult mlir::linalg_ext::ScanOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
  if (resultNumber == 1) {
    int64_t rank = getOperandRank();
    if (rank > 1) {
      for (auto i : llvm::seq<int64_t>(0, rank)) {
        if (i == getDimension())
          continue;
        resultOffsets.push_back(offsets[i]);
        resultSizes.push_back(sizes[i]);
      }
    }
    return success();
  }
  return failure();
}

//////////////////////////////
// SoftmaxOp
//////////////////////////////

mlir::LogicalResult mlir::linalg_ext::SoftmaxOp::verify() {
  Operation *op = getOperation();
  if (getNumOutputs() != 4) {
    return op->emitOpError("expected 4 output operands");
  }
  return success();
}

SmallVector<utils::IteratorType>
mlir::linalg_ext::SoftmaxOp::getLoopIteratorTypes() {
  // All loops except the dimension are parallel.
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

ArrayAttr mlir::linalg_ext::SoftmaxOp::getIndexingMaps() {
  unsigned rank = getOperandRank();
  SmallVector<AffineMap> maps;
  MLIRContext *ctx = getContext();

  // input
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));

  // outputs
  // result
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
  unsigned dim = getDimension();
  // scale
  maps.push_back(getMultiDimIdentityMapWithSkip(rank, dim, ctx));
  // max
  maps.push_back(getMultiDimIdentityMapWithSkip(rank, dim, ctx));
  // accum
  maps.push_back(getMultiDimIdentityMapWithSkip(rank, dim, ctx));

  return Builder(ctx).getAffineMapArrayAttr(maps);
}

SmallVector<Range> mlir::linalg_ext::SoftmaxOp::getIterationDomain(
    class mlir::OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> ranges;
  for (auto dim : llvm::seq<int64_t>(0, getOperandRank())) {
    Value ub = getDimValue(builder, loc, getOutputs()[0], dim);
    ranges.emplace_back(Range{zero, ub, one});
  }
  return ranges;
}

SmallVector<Operation *> mlir::linalg_ext::SoftmaxOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getOperandRank();
  assert(offsets.size() == static_cast<size_t>(rank) &&
         sizes.size() == static_cast<size_t>(rank));

  auto oneAttr = builder.getI64IntegerAttr(1);

  SmallVector<OpFoldResult> strides(rank, oneAttr);

  Location loc = getLoc();
  SmallVector<Value> tiledOperands;

  // input // operand 0 // data
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), input(), offsets, sizes, strides));

  // output // operand 1 // result
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), getOutputs()[0], offsets, sizes, strides));
  if (rank > 1) {
    ////////////////////
    // handle scale
    ////////////////////
    SmallVector<OpFoldResult> scaleOffsets, scaleSizes;
    // use getResultTilePosition with index as 1 for scale, since they use the
    // same tile position
    if (failed(getResultTilePosition(builder, 1, offsets, sizes, scaleOffsets,
                                     scaleSizes))) {
      return {};
    }

    SmallVector<OpFoldResult> scaleStrides(rank - 1, oneAttr);
    // output // operand 2 // scale
    tiledOperands.emplace_back(getSlice(builder, getLoc(), getOutputs()[1],
                                        scaleOffsets, scaleSizes,
                                        scaleStrides));

    ////////////////////
    // handle max carry
    ////////////////////
    SmallVector<OpFoldResult> maxOffsets, maxSizes;
    // use getResultTilePosition with index as 2 for max, since they use the
    // same tile position
    if (failed(getResultTilePosition(builder, 2, offsets, sizes, maxOffsets,
                                     maxSizes))) {
      return {};
    }
    SmallVector<OpFoldResult> maxStrides(rank - 1, oneAttr);
    // output // operand 2 // accum loop carry
    tiledOperands.emplace_back(getSlice(builder, getLoc(), getOutputs()[2],
                                        maxOffsets, maxSizes, maxStrides));

    ////////////////////
    // handle accum carry
    ////////////////////
    SmallVector<OpFoldResult> accumOffsets, accumSizes;
    // use getResultTilePosition with index as 3 for accum, since they use the
    // same tile position
    if (failed(getResultTilePosition(builder, 3, offsets, sizes, accumOffsets,
                                     accumSizes))) {
      return {};
    }
    SmallVector<OpFoldResult> accumStrides(rank - 1, oneAttr);
    // output // operand 3 // accum loop carry
    tiledOperands.emplace_back(getSlice(builder, getLoc(), getOutputs()[3],
                                        accumOffsets, accumSizes,
                                        accumStrides));
  } else {
    tiledOperands.emplace_back(getOutputs()[1]);
    tiledOperands.emplace_back(getOutputs()[2]);
    tiledOperands.emplace_back(getOutputs()[3]);
  }

  SmallVector<Type, 4> resultTypes;
  if (hasTensorSemantics()) {
    resultTypes.push_back(tiledOperands[1].getType());
    resultTypes.push_back(tiledOperands[2].getType());
    resultTypes.push_back(tiledOperands[3].getType());
    resultTypes.push_back(tiledOperands[4].getType());
  }

  Operation *tiledSoftmaxOp =
      cast<DestinationStyleOpInterface>(getOperation())
          .clone(builder, loc, resultTypes, tiledOperands);
  return {tiledSoftmaxOp};
  //  return {};
}

LogicalResult mlir::linalg_ext::SoftmaxOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
  if (resultNumber == 1 || resultNumber == 2 || resultNumber == 3) {
    int64_t rank = getOperandRank();
    if (rank > 1) {
      for (auto i : llvm::seq<int64_t>(0, rank)) {
        if (i == getDimension())
          continue;
        resultOffsets.push_back(offsets[i]);
        resultSizes.push_back(sizes[i]);
      }
    }
    return success();
  }
  return failure();
}

#define DEFINE_OP_GET_EFFECTS(OP_NAME)                                         \
  void OP_NAME::getEffects(                                                    \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>      \
          &effects) {                                                          \
    SmallVector<Value> inputBuffers = getInputBufferOperands();                \
    SmallVector<Value> outputBuffers = getOutputBufferOperands();              \
    getEffectsImpl(effects, getOperation()->getResults(), inputBuffers,        \
                   outputBuffers);                                             \
  }

DEFINE_OP_GET_EFFECTS(CustomOp)
DEFINE_OP_GET_EFFECTS(ScanOp)
DEFINE_OP_GET_EFFECTS(SoftmaxOp)

static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto castOp = operand.get().getDefiningOp<memref::CastOp>();
    if (castOp && memref::CastOp::canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

#define DEFINE_OP_FOLD(OP_NAME)                                                \
  LogicalResult OP_NAME::fold(ArrayRef<Attribute>,                             \
                              SmallVectorImpl<OpFoldResult> &) {               \
    return foldMemRefCast(*this);                                              \
  }

DEFINE_OP_FOLD(ScanOp)
DEFINE_OP_FOLD(SoftmaxOp)

namespace {
/// This is derived from mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp without any
/// changes.
struct FoldTensorCastOp : public OpInterfaceRewritePattern<LinalgExtOp> {
  using OpInterfaceRewritePattern<LinalgExtOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgExtOp op,
                                PatternRewriter &rewriter) const override {
    // If no operand comes from a tensor::CastOp and can be folded then fail.
    bool hasTensorCastOperand =
        llvm::any_of(op.getInputAndOutputOperands(), [&](OpOperand *opOperand) {
          if (opOperand->get().isa<BlockArgument>())
            return false;
          auto castOp = opOperand->get().getDefiningOp<tensor::CastOp>();
          return castOp && canFoldIntoConsumerOp(castOp);
        });
    if (!hasTensorCastOperand)
      return failure();

    SmallVector<Type, 4> newResultTypes;
    newResultTypes.reserve(op->getNumResults());
    SmallVector<Value, 4> newOperands;
    newOperands.reserve(op->getNumOperands());
    // Inputs may fold.
    for (OpOperand *opOperand : op.getInputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      newOperands.push_back(canFoldIntoConsumerOp(tensorCastOp)
                                ? tensorCastOp.getSource()
                                : opOperand->get());
    }
    // Init tensors may fold, in which case the resultType must also change.
    for (OpOperand *opOperand : op.getOutputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      bool fold = canFoldIntoConsumerOp(tensorCastOp);
      newOperands.push_back(fold ? tensorCastOp.getOperand()
                                 : opOperand->get());
      newResultTypes.push_back(newOperands.back().getType());
    }
    // Add the other operands.
    for (OpOperand *opOperand : op.getNonInputOrOutputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      newOperands.push_back(canFoldIntoConsumerOp(tensorCastOp)
                                ? tensorCastOp.getSource()
                                : opOperand->get());
    }
    // Clone op.
    Operation *newOp =
        op.clone(rewriter, op->getLoc(), newResultTypes, newOperands);
    SmallVector<Value, 4> replacements;
    replacements.reserve(newOp->getNumResults());
    for (auto result : llvm::zip(op->getResults(), newOp->getResults())) {
      Value oldResult = std::get<0>(result);
      Value newResult = std::get<1>(result);
      if (newResult.getType() != oldResult.getType()) {
        replacements.push_back(rewriter.create<tensor::CastOp>(
            op->getLoc(), oldResult.getType(), newResult));
      } else {
        replacements.push_back(newResult);
      }
    }
    rewriter.replaceOp(op, replacements);

    return success();
  }
};

} // namespace

void mlir::linalg_ext::LinalgExtDialect::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results) const {
  results.add<FoldTensorCastOp>(getContext());
}

#define GET_OP_CLASSES
#include "byteir/Dialect/Linalg/IR/LinalgExtOps.cpp.inc"
