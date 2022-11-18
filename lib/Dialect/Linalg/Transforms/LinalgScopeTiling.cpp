//===- LinalgScopeTiling.cpp -----------------------------------*--- C++
//-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

// use some code from Linalg's LinalgTiling.cpp

#include "byteir/Dialect/Linalg/Transforms/LinalgScopeTiling.h"
#include "byteir/Dialect/Linalg/Transforms/TilingUtils.h"
#include "byteir/Utils/Hoist.h"
#include "byteir/Utils/MemUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::linalg;
using namespace mlir::scf;

#define DEBUG_TYPE "linalg-scope-tiling"

namespace {

struct DimProperty {
  constexpr static int NoDim = -1;

  int dim;
  bool performedReduce;

  DimProperty() : dim(-1), performedReduce(false) {}

  DimProperty(int d, bool reduced) : dim(d), performedReduce(reduced) {}
};

// creat a single range
static SmallVector<Range, 4>
makeTiledLoopRange(OpBuilder &b, Location loc, AffineMap map,
                   ValueRange allShapeSizes, unsigned axis, int64_t tileSize) {
  assert(axis < map.getNumResults());

  // Apply `map` to get shape sizes in loop order.
  auto shapeSizes = applyMapToValues(b, loc, map, allShapeSizes);
  Value tileSizeValue = b.create<ConstantIndexOp>(loc, tileSize);

  // Create a new range with the applied tile sizes.
  SmallVector<Range, 4> res;
  res.push_back(Range{b.create<ConstantIndexOp>(loc, 0).getValue(),
                      shapeSizes[axis], tileSizeValue});
  return res;
}

static SmallVector<OpFoldResult, 4> createTileSize(OpBuilder &b, Location loc,
                                                   unsigned axis, unsigned rank,
                                                   int64_t tileSize) {

  SmallVector<OpFoldResult, 4> tileSizes;
  for (unsigned i = 0; i < rank; ++i) {
    int64_t val = i == axis ? tileSize : 0;
    tileSizes.push_back(b.create<ConstantIndexOp>(loc, val).getValue());
  }
  return tileSizes;
}

static bool isReduction(mlir::linalg::LinalgOp linalgOp) {
  MLIRContext *ctx = linalgOp.getContext();
  SmallVector<StringAttr> iterTypes = llvm::to_vector<4>(
      llvm::map_range(linalgOp.getIteratorTypesArray(),
                      [=](StringRef s) { return StringAttr::get(ctx, s); }));
  unsigned axis =
      linalgOp->getAttrOfType<IntegerAttr>(getScopeTilingAxisAttrName())
          .getInt();
  return isReductionIterator(iterTypes[axis]);
}

void tileScopeImpl(OpBuilder &b, TileScope &ts, int64_t tileSize,
                   bool parallelizeReduction,
                   LinalgScopeTilingLoopType loopType) {
  // early termination
  if (ts.ops.size() == 0)
    return;

  // 1. Build the tiled loop ranges.
  //    Use lastop to create loops variables
  auto lastOp = ts.ops.back();
  unsigned lastAxis =
      lastOp->getAttrOfType<IntegerAttr>(getScopeTilingAxisAttrName()).getInt();
  auto loc = lastOp.getLoc();

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(lastOp);

  auto lastAllShapeSizes =
      getAsValues(b, loc, lastOp.createFlatListOfOperandDims(b, loc));

  AffineMap lastShapeSizesToLoopsMap = lastOp.getShapesToLoopsMap();
  if (!lastShapeSizesToLoopsMap)
    return;

  SmallVector<Range, 4> loopRanges = makeTiledLoopRange(
      b, loc, lastShapeSizesToLoopsMap, lastAllShapeSizes, lastAxis, tileSize);

  // iteratorTypes Attribute
  SmallVector<Attribute, 4> iteratorTypes;
  iteratorTypes.push_back(
      b.getStringAttr(lastOp.getIteratorTypesArray()[lastAxis]));

  // If interchangeVector is empty, use the identity. Build the permutation map
  // otherwise.
  // TODO support loop interchange later

  bool isParallel = true;

  for (auto &linalgOp : ts.ops) {
    if (isReduction(linalgOp)) {
      isParallel = false;
      break;
    }
  }

  // 2. Create the tiled loops.
  SmallVector<Value, 8> lbsStorage, ubsStorage, stepsStorage;
  unpackRanges(b, loc, loopRanges, lbsStorage, ubsStorage, stepsStorage);
  ValueRange lbs(lbsStorage), ubs(ubsStorage), steps(stepsStorage);
  SmallVector<Value, 8> ivs(lbs.size());

  auto getInputBufferOperands = [](LinalgOp linalgOp) {
    OpOperandVector result;
    int64_t numInputs = linalgOp.getNumDpsInputs();
    result.reserve(numInputs);
    llvm::transform(
        linalgOp.getOperation()->getOpOperands().take_front(numInputs),
        std::back_inserter(result),
        [](OpOperand &opOperand) { return &opOperand; });
    return result;
  };

  auto tiledLoopBodyBuilder = [&](OpBuilder &b, Location loc,
                                  ValueRange loopIvs) {
    ivs.assign(loopIvs.begin(), loopIvs.end());

    // When an `interchangeVector` is present, it has been applied to the
    // loop ranges and the iterator types. Apply its inverse to the
    // resulting loop `ivs` to match the op definition.
    SmallVector<OpFoldResult, 4> interchangedIvs;

    // TODO support loop interchange later
    interchangedIvs.assign(ivs.begin(), ivs.end());

    // go through all ops
    for (auto &linalgOp : ts.ops) {

      DestinationStyleOpInterface dstStyleOp =
          cast<DestinationStyleOpInterface>(linalgOp.getOperation());

      SmallVector<OpOperand *> outputBufferOperands, outputTensorOperands;
      for (OpOperand *operand : dstStyleOp.getDpsInitOperands()) {
        Type type = operand->get().getType();
        if (type.isa<MemRefType>())
          outputBufferOperands.push_back(operand);
        if (type.isa<RankedTensorType>())
          outputTensorOperands.push_back(operand);
      }

      unsigned axis =
          linalgOp->getAttrOfType<IntegerAttr>(getScopeTilingAxisAttrName())
              .getInt();
      unsigned rank =
          linalgOp->getAttrOfType<IntegerAttr>(getScopeTilingRankAttrName())
              .getInt();

      auto localAllShapeSizes =
          getAsValues(b, loc, linalgOp.createFlatListOfOperandDims(b, loc));
      AffineMap localShapeSizesToLoopsMap = linalgOp.getShapesToLoopsMap();

      auto localSizeBoundsOfValues = applyMapToValues(
          b, loc, localShapeSizesToLoopsMap, localAllShapeSizes);
      SmallVector<OpFoldResult> localSizeBounds(localSizeBoundsOfValues.begin(),
                                                localSizeBoundsOfValues.end());

      // get all values for now
      // TODO: relax this later
      SmallVector<Value> valuesToTile = linalgOp->getOperands();

      SmallVector<OpFoldResult, 4> tileSizes =
          createTileSize(b, loc, axis, rank, tileSize);

      SmallVector<Value, 4> localTiledOperands = makeTiledShapes(
          b, loc, linalgOp, valuesToTile, interchangedIvs, tileSizes,
          localSizeBounds, /*omitPartialTileCheck=*/false);

      SmallVector<Type, 4> resultTensorTypes;
      for (OpOperand *opOperand : outputTensorOperands) {
        resultTensorTypes.push_back(
            localTiledOperands[opOperand->getOperandNumber()].getType());
      }

      // if enabling parallelize reduction and the loop is reduction
      // we need to alloc a new buffer and perform all reduce
      if (parallelizeReduction && isReduction(linalgOp)) {
        SmallVector<Value> intermediates;
        SmallVector<Value> outputs;
        for (size_t i = getInputBufferOperands(linalgOp).size();
             i < localTiledOperands.size(); ++i) {
          outputs.push_back(localTiledOperands[i]); // record all one
          auto maybeValue = createAlloc(b, localTiledOperands[i]);
          intermediates.push_back(maybeValue.value());
          localTiledOperands[i] = maybeValue.value();
        }

        auto cloned =
            linalgOp.clone(b, loc, resultTensorTypes, localTiledOperands);

        // remove attr
        cloned->removeAttr(getScopeTilingAxisAttrName());
        cloned->removeAttr(getScopeTilingRankAttrName());

        // support arith::AtomicRMWKind::addf for now
        // FIXME: extend it
        auto maybeAlloc = createAtomicLinalgGeneric(
            b, loc, arith::AtomicRMWKind::addf, intermediates, outputs);

        if (!maybeAlloc.has_value())
          return;
      } else {
        auto cloned =
            linalgOp.clone(b, loc, resultTensorTypes, localTiledOperands);

        // remove attr
        cloned->removeAttr(getScopeTilingAxisAttrName());
        cloned->removeAttr(getScopeTilingRankAttrName());
      }
    }
  };

  if (parallelizeReduction)
    isParallel = true;

  if (loopType == LinalgScopeTilingLoopType::SCFLoops) {
    if (failed(buildSCFLoop(b, loc, isParallel, lbs.take_front(),
                            ubs.take_front(), steps.take_front(),
                            tiledLoopBodyBuilder))) {
      return;
    }
  } else if (loopType == LinalgScopeTilingLoopType::AffineLoops) {
    // affine not support parallelizeReduction for now
    if (parallelizeReduction)
      return;

    if (failed(buildAffineLoop(b, loc, lbs.take_front(), ubs.take_front(),
                               steps.take_front(), tiledLoopBodyBuilder))) {
      return;
    }
  } else {
    // TODO support Linalg::TiledLoop later
    return;
  }

  for (auto &op : ts.ops) {
    op->erase();
  }
}

/**
 * find iteration index through dim and inversePermutation
 * E.g. if affineMap = (d0, d1, d2)-> (d0, d2), dim = 1
 * Then invMap = (d0, d1)->(d0, 0, d1)
 *      OneHot = (0, 1)
 *      invComposed = (0, 0, 1)
 *      iterAxis = 2
 */
static Optional<unsigned> getIterAxis(AffineMap affineMap, unsigned dimIndex) {
  // AffineMap invMap = inversePermutation(affineMap);
  AffineMap invMap = inverseAndBroadcastProjectedPermutation(affineMap);
  if (invMap.isEmpty())
    return llvm::None;
  auto invComposed =
      invMap.compose(createOneHot(invMap.getNumInputs(), dimIndex));
  auto iterAxes = getAllIndicesForNonZeros(invComposed);
  // no support all-to-1 or non mapping
  if (iterAxes.size() != 1) {
    return llvm::None;
  }
  return iterAxes[0];
}

// update view to DimAndOps mapping
static void
updateViewToDimAndOps(llvm::DenseMap<Value, DimProperty> &viewToDimProp,
                      Value view, LinalgOp op, AffineMap &affineMap,
                      unsigned iterAxis, bool reduced, bool assigned = false) {

  if (viewToDimProp.count(view) == 0) {
    auto composed =
        affineMap.compose(createOneHot(affineMap.getNumDims(), iterAxis));
    auto dimAxes = getAllIndicesForNonZeros(composed);

    // only handle one with 1 valid dimAxes
    if (dimAxes.size() == 1) {
      viewToDimProp[view] = DimProperty(dimAxes[0], reduced);
    } else if (assigned && dimAxes.size() == 0) {
      viewToDimProp[view] = DimProperty(DimProperty::NoDim, reduced);
    }
  }
}

static bool isProducerValidforTileScope(
    llvm::DenseMap<Value, DimProperty> &viewToDimAndOps,
    llvm::DenseMap<Operation *, std::pair<unsigned, unsigned>>
        &opToIterAxisAndRank,
    Operation *op) {
  if (!isa<LinalgOp>(op)) {
    return MemoryEffectOpInterface::hasNoEffect(op);
  }

  auto producer = cast<LinalgOp>(op);
  // only handle strucutre op
  if (!isStructuralLinalg(producer)) {
    return false;
  }

  unsigned iterAxis;
  int64_t numInputs = producer.getNumDpsInputs();

  SmallVector<AffineMap> affineMaps = llvm::to_vector<4>(
      producer.getIndexingMaps().getAsValueRange<AffineMapAttr>());

  if (opToIterAxisAndRank.count(op) == 0) {
    auto producerOutView = producer.getDpsInitOperand(0)->get();
    if (viewToDimAndOps.count(producerOutView) == 0) {
      // no fuse non-producer-consumer op
      return false;
    }
    int outDimIndex = viewToDimAndOps[producerOutView].dim;
    if (outDimIndex == DimProperty::NoDim)
      return false;
    auto maybeIterAxis = getIterAxis(affineMaps[numInputs], outDimIndex);
    if (!maybeIterAxis.has_value())
      return false;
    iterAxis = maybeIterAxis.value();
    opToIterAxisAndRank[op] = {iterAxis, affineMaps[numInputs].getNumDims()};
  } else {
    iterAxis = opToIterAxisAndRank[op].first;
  }

  MLIRContext *ctx = op->getContext();
  SmallVector<StringAttr> iterTypes = llvm::to_vector<4>(
      llvm::map_range(producer.getIteratorTypesArray(),
                      [=](StringRef s) { return StringAttr::get(ctx, s); }));
  // For now, avoid fusion along reduction iterator
  if (isReductionIterator(iterTypes[iterAxis]))
    return false;

  auto iterOneHot = createOneHot(iterTypes.size(), iterAxis);

  // insert producer's input
  for (int64_t i = 0; i < numInputs; ++i) {
    updateViewToDimAndOps(viewToDimAndOps,
                          producer.getDpsInputOperand(i)->get(), producer,
                          affineMaps[i], iterAxis, false /*performedReduce*/);
  }

  return true;
}

static bool isConsumerValidforTileScope(
    llvm::DenseMap<Value, DimProperty> &viewToDimProp,
    llvm::DenseMap<Operation *, std::pair<unsigned, unsigned>>
        &opToIterAxisAndRank,
    Operation *op) {
  if (!isa<LinalgOp>(op)) {
    return MemoryEffectOpInterface::hasNoEffect(op);
  }

  auto consumer = cast<LinalgOp>(op);

  // only handle strucutral op
  if (!isStructuralLinalg(consumer)) {
    return false;
  }

  SmallVector<AffineMap> affineMaps = llvm::to_vector<4>(
      consumer.getIndexingMaps().getAsValueRange<AffineMapAttr>());
  int64_t numInputs = consumer.getNumDpsInputs();
  unsigned iterAxis;

  if (opToIterAxisAndRank.count(op) == 0) {

    bool atLeastOne = false;

    for (int64_t i = 0; i < numInputs; ++i) {
      auto input = consumer.getDpsInputOperand(i)->get();
      if (viewToDimProp.count(input) != 0) {
        int intDimIndex = viewToDimProp[input].dim;

        if (intDimIndex == DimProperty::NoDim) {
          return false;
        }
        auto inferredIterAxis = getIterAxis(affineMaps[i], intDimIndex);
        if (!inferredIterAxis.has_value())
          continue;
        if (!atLeastOne) {
          iterAxis = inferredIterAxis.value();
        } else if (iterAxis != inferredIterAxis.value()) {
          return false;
        }
        atLeastOne = true;
      }
    }

    if (!atLeastOne) {
      return false;
    }

    opToIterAxisAndRank[op] = {iterAxis, affineMaps[numInputs].getNumDims()};
  } else {
    iterAxis = opToIterAxisAndRank[op].first;
  }

  MLIRContext *ctx = op->getContext();
  SmallVector<StringAttr> iterTypes = llvm::to_vector<4>(
      llvm::map_range(consumer.getIteratorTypesArray(),
                      [=](StringRef s) { return StringAttr::get(ctx, s); }));

  bool performedReduced = isReductionIterator(iterTypes[iterAxis]);

  // insert consumer's output
  {
    updateViewToDimAndOps(viewToDimProp, consumer.getDpsInitOperand(0)->get(),
                          consumer, affineMaps[numInputs], iterAxis,
                          performedReduced, true /*assigned*/);
  }

  // insert consumer' other input
  for (int64_t i = 0; i < numInputs; ++i) {
    updateViewToDimAndOps(viewToDimProp, consumer.getDpsInputOperand(i)->get(),
                          consumer, affineMaps[i], iterAxis,
                          false /*performedReduced*/);
  }

  return true;
}

// Collect ops within the Block of a given AnchorOp
static void greedilyCreateTilingScopeFromAnchor(TileScope &ts,
                                                unsigned iterAxis) {
  LinalgOp anchorOp = ts.anchorOp;
  Block *block = anchorOp->getBlock();

  // id to ops
  SmallVector<Operation *> ops;
  int anchorId = -1;
  // initialization
  for (auto &op : *block) {
    if (anchorId == -1 && anchorOp.getOperation() == &op) {
      anchorId = static_cast<int>(ops.size());
    }
    ops.push_back(&op);
  }

  SmallVector<AffineMap> affineMaps = llvm::to_vector<4>(
      anchorOp.getIndexingMaps().getAsValueRange<AffineMapAttr>());

  MLIRContext *ctx = anchorOp.getContext();
  SmallVector<StringAttr> iterTypes = llvm::to_vector<4>(
      llvm::map_range(anchorOp.getIteratorTypesArray(),
                      [=](StringRef s) { return StringAttr::get(ctx, s); }));

  // if iterAxis is out of bound, early terminate
  if (iterAxis >= iterTypes.size()) {
    return;
  }

  llvm::DenseMap<Value, DimProperty> viewToDimProp;
  llvm::DenseMap<Operation *, std::pair<unsigned, unsigned>>
      opToIterAxisAndRank;

  opToIterAxisAndRank[anchorOp] = {
      iterAxis, static_cast<unsigned>(affineMaps.back().getNumDims())};
  bool performedReduce = isReductionIterator(iterTypes[iterAxis]);

  // insert producer's input
  int64_t numInputs = anchorOp.getNumDpsInputs();
  for (int64_t i = 0; i < numInputs; ++i) {
    updateViewToDimAndOps(viewToDimProp, anchorOp.getDpsInputOperand(i)->get(),
                          anchorOp, affineMaps[i], iterAxis,
                          false /*performedReduce*/);
  }

  int64_t numOutputs = anchorOp.getNumDpsInits();
  for (int64_t i = 0; i < numOutputs; ++i) {
    updateViewToDimAndOps(viewToDimProp, anchorOp.getDpsInitOperand(i)->get(),
                          anchorOp, affineMaps[numInputs + i], iterAxis,
                          performedReduce, true /*assigned*/);
  }

  // NOTE: use set here, instead of SmallSet, since we need ids to be sorted.
  std::set<int> determinedIds;
  determinedIds.insert(anchorId);

  int iterCnt = 0;
  while (iterCnt++ < static_cast<int>(block->getOperations().size())) {
    bool changed = false;
    // producer
    for (int i = anchorId - 1; i >= 0; --i) {
      if (determinedIds.count(i) > 0)
        continue;
      if (isProducerValidforTileScope(viewToDimProp, opToIterAxisAndRank,
                                      ops[i])) {
        changed = true;
        determinedIds.insert(i);
      } else {
        break;
      }
    }
    // consumer
    for (int i = anchorId + 1; i < static_cast<int>(ops.size()); ++i) {
      if (determinedIds.count(i) > 0)
        continue;
      if (isConsumerValidforTileScope(viewToDimProp, opToIterAxisAndRank,
                                      ops[i])) {
        changed = true;
        determinedIds.insert(i);
      } else {
        break;
      }
    }
    if (!changed)
      break;
  }

  // insert attr for scope tiling
  for (auto id : determinedIds) {
    if (auto linalg = dyn_cast<LinalgOp>(ops[id])) {

      linalg->setAttr(getScopeTilingAxisAttrName(),
                      IntegerAttr::get(IntegerType::get(ctx, 32),
                                       opToIterAxisAndRank[ops[id]].first));

      linalg->setAttr(getScopeTilingRankAttrName(),
                      IntegerAttr::get(IntegerType::get(ctx, 32),
                                       opToIterAxisAndRank[ops[id]].second));

      ts.ops.push_back(linalg);
    }
  }
}

static void collectTilingScope(func::FuncOp func, unsigned iterAxis,
                               SmallVectorImpl<TileScope> &collection) {

  SmallSet<Block *, 4> visitedBlocks;

  // collect op with anchor as intial values
  // it is nested
  func.walk([&](LinalgOp op) {
    // skip non-targeting or visited block
    if (!op->hasAttr(getScopeTilingAnchorAttrName()) ||
        visitedBlocks.contains(op->getBlock())) {
      return;
    }

    op->removeAttr(getScopeTilingAnchorAttrName());
    collection.emplace_back(op);
    visitedBlocks.insert(op->getBlock());
  });

  // collect ops with ScopeTilingAxisAttr & ScopeTilingRankAttr
  llvm::SmallDenseMap<Block *, SmallVector<LinalgOp>> blockToScope;
  llvm::SmallSet<Block *, 4> noParallel;

  func.walk([&](LinalgOp op) {
    Block *block = op->getBlock();
    // skip non-targeting or visited block
    if (!op->hasAttr(getScopeTilingAxisAttrName()) ||
        !op->hasAttr(getScopeTilingRankAttrName()) ||
        visitedBlocks.contains(block)) {
      return;
    }

    blockToScope[block].push_back(op);
  });

  // create tiling scope for each anchorTag
  for (TileScope &ts : collection) {
    greedilyCreateTilingScopeFromAnchor(ts, iterAxis);
  }

  for (auto &it : blockToScope) {
    TileScope ts(it.second.front()); // use the first as anchor
    ts.ops = it.second;              // copy
    collection.push_back(ts);
  }
}

bool isHoistUpOp(Operation *op) {
  return isa<memref::AllocOp, memref::CollapseShapeOp, memref::DimOp,
             memref::ExpandShapeOp, memref::ReshapeOp>(op);
}

bool isHoistDownOp(Operation *op) { return isa<memref::DeallocOp>(op); }

struct LinalgScopeTilingPass
    : public LinalgScopeTilingBase<LinalgScopeTilingPass> {
  LinalgScopeTilingPass() = default;
  LinalgScopeTilingPass(int64_t tileAxis, int64_t tileSize,
                        bool parallelizeReduction,
                        LinalgScopeTilingLoopType loopType,
                        StringRef distributionType) {

    this->tileAxis = tileAxis;
    this->tileSize = tileSize;
    this->parallelizeReduction = parallelizeReduction;
    this->loopType = "";
    loopTypeEnum = loopType;
    this->distributionType = distributionType.str();
  }

  void runOnOperation() override {
    // early terminate when tileSize == 0
    if (tileSize == 0)
      return;

    // parse
    LinalgScopeTilingLoopType type =
        llvm::StringSwitch<LinalgScopeTilingLoopType>(loopType)
            .Case("scf", LinalgScopeTilingLoopType::SCFLoops)
            .Case("affine", LinalgScopeTilingLoopType::AffineLoops)
            .Case("tiled_loop", LinalgScopeTilingLoopType::TiledLoops)
            .Default(loopTypeEnum);

    func::FuncOp funcOp = getOperation();

    auto &domInfo = getAnalysis<DominanceInfo>();
    auto &postDomInfo = getAnalysis<PostDominanceInfo>();

    // hoisting
    for (auto &block : funcOp.getBody()) {
      hoistUpOpsInBlock(&block, domInfo, isHoistUpOp);
      hoistDownOpsInBlock(&block, postDomInfo, isHoistDownOp);
    }

    SmallVector<TileScope> collection;
    collectTilingScope(funcOp, tileAxis, collection);

    OpBuilder b(funcOp.getContext());
    for (auto &ts : collection) {
      tileScopeImpl(b, ts, tileSize, parallelizeReduction, type);
    }
  }

  LinalgScopeTilingLoopType loopTypeEnum;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgScopeTilingPass(
    int64_t tileAxis, int64_t tileSize, bool parallelizeReduction,
    mlir::LinalgScopeTilingLoopType loopType, StringRef distributionType) {
  return std::make_unique<LinalgScopeTilingPass>(
      tileAxis, tileSize, parallelizeReduction, loopType, distributionType);
}
