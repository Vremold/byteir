//===- LinalgExtOps.cpp ---------------------------------------------------===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
// Some code comes from LinalgOps.cpp in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Utils/AffineUtils.h"
#include "byteir/Utils/Utils.h"
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
// AliasOp
//////////////////////////////

mlir::LogicalResult mlir::linalg_ext::AliasOp::verify() {
  auto op = getOperation();
  if (op->getOperand(0).getType() != op->getResult(0).getType()) {
    return op->emitOpError("expected same type of operand and result");
  }
  return success();
}

//////////////////////////////
// CustomOp
//////////////////////////////

mlir::LogicalResult mlir::linalg_ext::CustomOp::verify() {
  // FIXME
  return success();
}

FailureOr<Value> mlir::linalg_ext::CustomOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  // FIXME
  return failure();
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
// DiagOp
//////////////////////////////

Type mlir::linalg_ext::DiagOp::getDiagType(ShapedType type) {
  auto shape = type.getShape();
  SmallVector<int64_t> newShape(shape.begin(), shape.end());
  newShape.insert(newShape.end(), shape.begin(), shape.end());
  return type.clone(newShape);
}

mlir::LogicalResult mlir::linalg_ext::DiagOp::verify() {
  // FIXME
  return success();
}

ArrayAttr mlir::linalg_ext::DiagOp::getIndexingMaps() {
  unsigned rank = getOperandRank();
  SmallVector<AffineMap> maps;
  MLIRContext *ctx = getContext();

  // input
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank / 2, ctx));

  // outputs
  // result
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
  return Builder(ctx).getAffineMapArrayAttr(maps);
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

FailureOr<Value> mlir::linalg_ext::ScanOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  // FIXME
  return failure();
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

namespace {
/// Return success if involved iterAxes includes dim,
/// Return failure otherwise
/// TODO move this to public
bool involveReduction(Operation &op, ArrayRef<mlir::AffineMap> indexingMaps,
                      ArrayRef<utils::IteratorType> loopIteratorTypes) {
  for (const auto &en : llvm::enumerate(op.getOperands())) {
    llvm::SmallVector<::mlir::OpFoldResult, 4> mixedOffsets;

    if (auto sliceOp = en.value().getDefiningOp<tensor::ExtractSliceOp>()) {
      mixedOffsets = sliceOp.getMixedOffsets();
    } else if (auto subviewOp = en.value().getDefiningOp<memref::SubViewOp>()) {
      mixedOffsets = subviewOp.getMixedOffsets();
    } else {
      continue;
    }

    auto indexingMap = indexingMaps[en.index()];
    for (const auto &en2 : llvm::enumerate(mixedOffsets)) {
      auto value = en2.value().dyn_cast<Value>();
      if (!value) {
        // since not a value, it implies not a loop arg
        continue;
      }

      auto iterArg = value.dyn_cast<BlockArgument>();
      if (!iterArg || !isa<scf::ForOp>(iterArg.getOwner()->getParentOp())) {
        // since not a BlockArgument or owner is a loop,
        // it implies not a loop arg
        continue;
      }

      FailureOr<unsigned> iterAxis =
          getIterAxisFromDim(indexingMap, en2.index());

      if (failed(iterAxis))
        continue;

      if (loopIteratorTypes[iterAxis.value()] ==
          utils::IteratorType::reduction) {
        return true;
      }
    }
  }
  return false;
}

mlir::LogicalResult validSoftmaxConsumer(Operation *op) {
  if (op == nullptr)
    return failure();

  // support matmul op now
  // TODO we will relax it
  if (isa<linalg::MatmulOp>(op)) {
    return success();
  }
  return failure();
}

Value getSoftmaxScaleDiagMatmul(OpBuilder &b, mlir::Location loc,
                                SoftmaxOp softmax, Value consumerOutput) {
  auto scale = softmax->getResult(3);
  if (auto scaleTensorTy = scale.getType().dyn_cast<TensorType>()) {
    if (!consumerOutput.getType().isa<TensorType>()) {
      // Not support mixing TensorType with other types
      return Value();
    }
    auto consumerTensorTy = consumerOutput.getType().cast<TensorType>();
    auto scaleEmpty = b.create<tensor::EmptyOp>(
        loc, DiagOp::getDiagType(scaleTensorTy), ValueRange{});
    auto diag =
        b.create<linalg_ext::DiagOp>(loc, scale, scaleEmpty.getResult());
    auto consumerEmpty =
        b.create<tensor::EmptyOp>(loc, consumerTensorTy, ValueRange{});

    SmallVector<Value> scaleMatmulInputs;
    scaleMatmulInputs.push_back(diag->getResult(0));
    scaleMatmulInputs.push_back(consumerOutput);
    auto scaleMatmul = b.create<linalg::MatmulOp>(loc, scaleMatmulInputs,
                                                  consumerEmpty->getResults());

    return scaleMatmul->getResult(0);
  }

  return Value();
}

void rewriteSoftmaxFusedConsumer(OpBuilder &b, SoftmaxOp fused, int64_t offset,
                                 Operation *consumer) {
  if (consumer == nullptr)
    return;

  auto result = fused.getResult(offset);
  b.setInsertionPoint(consumer);
  auto loc = consumer->getLoc();
  if (auto linaglOp = dyn_cast<linalg::LinalgOp>(consumer)) {
    // Here assume first ouput is fused as result
    // TODO: fix this if the assumption not hold
    auto firstOutput = linaglOp.getDpsInitOperand(0)->get();
    auto scaleMatmul = getSoftmaxScaleDiagMatmul(b, loc, fused, firstOutput);
    if (scaleMatmul == nullptr)
      return;
    linaglOp.setDpsInitOperand(0, scaleMatmul);
  }
}

void rewriteSoftmaxFusedConsumers(OpBuilder &b, Operation *unfused,
                                  SoftmaxOp fused, int64_t offset) {
  if (unfused == nullptr)
    return;

  auto result = unfused->getResult(offset);
  for (auto user : result.getUsers()) {
    if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
      for (auto sliceUser : sliceOp.getResult().getUsers()) {
        rewriteSoftmaxFusedConsumer(b, fused, offset, sliceUser);
      }
    }
  }
}

mlir::LogicalResult validSoftmaxFusedConsumer(Operation *op) {
  if (op == nullptr)
    return failure();

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    auto tiledOp = cast<TilingInterface>(op);
    if (!involveReduction(*op, linalgOp.getIndexingMapsArray(),
                          tiledOp.getLoopIteratorTypes())) {
      return failure();
    }
    return validSoftmaxConsumer(op);
  }
  return failure();
}

/// This is too conservative
/// TODO extend this
mlir::LogicalResult checkSoftmaxConsumers(Operation *unfused, int64_t offset) {

  if (unfused == nullptr)
    return failure();

  // particular result given an offset
  for (const auto &opResult : unfused->getOpResults()) {
    if (opResult.getResultNumber() == offset) {
      if (useCount(opResult) != 2) {
        // 2 as 1 consumer before fused and 1 from fused but not replaced value
        // this might be too conservative
        return failure();
      }

      for (const auto &use : opResult.getUses()) {
        if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(use.getOwner())) {
          for (auto sliceUser : sliceOp.getResult().getUsers()) {
            if (failed(validSoftmaxFusedConsumer(sliceUser))) {
              return failure();
            }
          }
        } else if (failed(validSoftmaxConsumer(use.getOwner()))) {
          return failure();
        }
      }
    } else {
      if (useCount(opResult) != 0) {
        // this might be too conservative
        return failure();
      }
    } // if opResult.getResultNumber() == offset
  }   // for opResult : unfused->getOpResults())

  return success();
}

} // namespace

mlir::LogicalResult mlir::linalg_ext::SoftmaxOp::verify() {
  Operation *op = getOperation();
  if (getNumInputs() != 1) {
    return op->emitOpError("expected one input operands");
  }
  if (getNumOutputs() != 4) {
    return op->emitOpError("expected 4 output operands");
  }
  if (!input().getType().isa<ShapedType>()) {
    return op->emitOpError("expected first input element type to be shaped");
  }

  auto maxType = max().getType().cast<ShapedType>();
  auto accumulatorType = accumulator().getType().cast<ShapedType>();
  auto scaleType = scale().getType().cast<ShapedType>();
  auto inputType = input().getType().cast<ShapedType>();
  auto outputType = output().getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShapes = inputType.getShape();
  ArrayRef<int64_t> outputShapes = outputType.getShape();
  if (maxType.getElementType() != inputType.getElementType() ||
      accumulatorType.getElementType() != inputType.getElementType() ||
      scaleType.getElementType() != inputType.getElementType() ||
      outputType.getElementType() != inputType.getElementType()) {
    return op->emitOpError("expected input/max/accumulator/scale/output "
                           "element types to be identical");
  }

  ArrayRef<int64_t> maxShape = maxType.getShape();
  int64_t maxRank = maxType.getRank();
  ArrayRef<int64_t> accumulatorShape = accumulatorType.getShape();
  int64_t accumulatorRank = accumulatorType.getRank();
  ArrayRef<int64_t> scaleShape = maxType.getShape();
  int64_t scaleRank = scaleType.getRank();
  int64_t expectedRank = inputType.getRank() - 1;
  if (maxRank != expectedRank || accumulatorRank != expectedRank ||
      scaleRank != expectedRank) {
    return op->emitOpError(
        "expected max/accumulator/scale rank to be equal to input rank - 1");
  }

  SmallVector<int64_t> expectedShape;
  for (int i = 0; i < inputType.getRank(); i++) {
    if (i != getDimension())
      expectedShape.push_back(inputShapes[i]);
  }
  if (llvm::any_of(
          llvm::zip(expectedShape, maxShape, accumulatorShape, scaleShape),
          [](std::tuple<int64_t, int64_t, int64_t, int64_t> s) {
            return std::get<0>(s) != ShapedType::kDynamicSize &&
                   std::get<1>(s) != ShapedType::kDynamicSize &&
                   std::get<2>(s) != ShapedType::kDynamicSize &&
                   std::get<3>(s) != ShapedType::kDynamicSize &&
                   std::get<0>(s) != std::get<1>(s) &&
                   std::get<0>(s) != std::get<2>(s) &&
                   std::get<0>(s) != std::get<3>(s);
          })) {
    return op->emitOpError("incompatible input/max/accumulator/scale shapes");
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

LogicalResult mlir::linalg_ext::SoftmaxOp::isValidTiling(Operation *tiled) {
  if (tiled == nullptr)
    return failure();
  if (involveReduction(*tiled, getIndexingMapsArray(),
                       getLoopIteratorTypes())) {
    return failure();
  }
  return success();
}

LogicalResult mlir::linalg_ext::SoftmaxOp::correctTiledConsumerOps(
    OpBuilder &b, Operation *fused, int64_t offset) {
  if (fused == nullptr)
    return failure();

  // check involving getDimension axis
  if (!involveReduction(*fused, getIndexingMapsArray(),
                        getLoopIteratorTypes())) {
    return failure();
  }

  auto op = getOperation();
  // check consumer
  if (failed(checkSoftmaxConsumers(op, offset))) {
    return failure();
  }

  // rewrite all fused consumers
  auto fusedSoftmax = cast<SoftmaxOp>(fused);
  rewriteSoftmaxFusedConsumers(b, op, fusedSoftmax, offset);

  return success();
}

bool mlir::linalg_ext::SoftmaxOp::isResultCleanable(int64_t number,
                                                    bool hasOneOrZeroUse,
                                                    bool allParallel) {
  assert(number < 4);

  if (number == 0) {
    return hasOneOrZeroUse;
  } else if (number == 3) {
    return true;
  } else if (number == 1 || number == 2) {
    return allParallel;
  }
  return false;
}

FailureOr<Value> mlir::linalg_ext::SoftmaxOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  auto op = getOperation();
  auto numLoops = getOperandRank();
  SmallVector<OpFoldResult> iterationTileOffsets(numLoops),
      iterationTileSizes(numLoops);

  auto indexingMaps =
      llvm::to_vector(getIndexingMaps().getAsValueRange<AffineMapAttr>());
  auto indexingMap = indexingMaps[1 + resultNumber]; // 1 from input

  if (!indexingMap.isProjectedPermutation()) {
    return op->emitOpError(
        "unhandled tiled implementation generation when result is not "
        "accessed using a permuted projection");
  }
  if (!indexingMap.isPermutation()) {
    SmallVector<Range> iterationDomain = getIterationDomain(b);
    for (const auto &range : llvm::enumerate(iterationDomain)) {
      iterationTileOffsets[range.index()] = range.value().offset;
      iterationTileSizes[range.index()] = range.value().size;
    }
  }
  for (const auto &resultExpr : llvm::enumerate(indexingMap.getResults())) {
    unsigned dimPosition =
        resultExpr.value().cast<AffineDimExpr>().getPosition();
    iterationTileOffsets[dimPosition] = offsets[resultExpr.index()];
    iterationTileSizes[dimPosition] = sizes[resultExpr.index()];
  }

  auto tilingInterfaceOp = cast<TilingInterface>(op);
  SmallVector<Operation *> tiledOp = tilingInterfaceOp.getTiledImplementation(
      b, iterationTileOffsets, iterationTileSizes);

  if (tiledOp.size() != 1)
    return op->emitOpError("failed to generate tiled implementation");

  return tiledOp[0]->getResult(resultNumber);
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
  // max
  maps.push_back(getMultiDimIdentityMapWithSkip(rank, dim, ctx));
  // accum
  maps.push_back(getMultiDimIdentityMapWithSkip(rank, dim, ctx));
  // scale
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
    // handle max carry
    ////////////////////
    SmallVector<OpFoldResult> maxOffsets, maxSizes;
    // use getResultTilePosition with index as 1 for max, since they use the
    // same tile position
    if (failed(getResultTilePosition(builder, 1, offsets, sizes, maxOffsets,
                                     maxSizes))) {
      return {};
    }
    SmallVector<OpFoldResult> maxStrides(rank - 1, oneAttr);
    // output // operand 1 // max loop carry
    tiledOperands.emplace_back(getSlice(builder, getLoc(), getOutputs()[1],
                                        maxOffsets, maxSizes, maxStrides));

    ////////////////////
    // handle accum carry
    ////////////////////
    SmallVector<OpFoldResult> accumOffsets, accumSizes;
    // use getResultTilePosition with index as 2 for accum, since they use the
    // same tile position
    if (failed(getResultTilePosition(builder, 2, offsets, sizes, accumOffsets,
                                     accumSizes))) {
      return {};
    }
    SmallVector<OpFoldResult> accumStrides(rank - 1, oneAttr);
    // output // operand 3 // accum loop carry
    tiledOperands.emplace_back(getSlice(builder, getLoc(), getOutputs()[2],
                                        accumOffsets, accumSizes,
                                        accumStrides));

    ////////////////////
    // handle scale
    ////////////////////
    SmallVector<OpFoldResult> scaleOffsets, scaleSizes;
    // use getResultTilePosition with index as 3 for scale, since they use the
    // same tile position
    if (failed(getResultTilePosition(builder, 3, offsets, sizes, scaleOffsets,
                                     scaleSizes))) {
      return {};
    }

    SmallVector<OpFoldResult> scaleStrides(rank - 1, oneAttr);
    // output // operand 2 // scale
    tiledOperands.emplace_back(getSlice(builder, getLoc(), getOutputs()[3],
                                        scaleOffsets, scaleSizes,
                                        scaleStrides));
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
DEFINE_OP_GET_EFFECTS(DiagOp)
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

DEFINE_OP_FOLD(DiagOp)
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
