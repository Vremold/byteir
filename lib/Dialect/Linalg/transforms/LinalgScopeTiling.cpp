//===- LinalgScopeTiling.cpp -----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

// use some code from Linalg's LinalgTiling.cpp

#include "byteir/Dialect/Linalg/transforms/LinalgScopeTiling.h"
#include "byteir/Dialect/Linalg/transforms/TilingUtils.h"
#include "byteir/Utils/Hoist.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"

#include "PassDetail.h"

#include <iostream> //debug

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::linalg;
using namespace mlir::scf;

#define DEBUG_TYPE "linalg-scope-tiling"

namespace {

struct DimProperty {
  constexpr static int  NoDim = -1;

  int dim;
  bool performedReduce;

  DimProperty()
    : dim(-1), performedReduce(false) {}

  DimProperty(int dim_, bool reduced)
    : dim(dim_), performedReduce(reduced) {}
};

// creat a single range
static SmallVector<Range, 4> makeTiledLoopRange(
  OpBuilder& b, Location loc, AffineMap map,
  ValueRange allShapeSizes, unsigned axis, int64_t tileSize) {
  assert(axis < map.getNumResults());

  // Apply `map` to get shape sizes in loop order.
  auto shapeSizes = applyMapToValues(b, loc, map, allShapeSizes);
  Value tileSizeValue = b.create<ConstantIndexOp>(loc, tileSize);
  
  // Create a new range with the applied tile sizes.
  SmallVector<Range, 4> res;
  res.push_back(Range{ b.create<ConstantIndexOp>(loc, 0), shapeSizes[axis], tileSizeValue});
  return res;
}

static SmallVector<Value, 4> createTileSize(
  OpBuilder& b,
  Location loc,
  unsigned axis,
  unsigned rank,
  int64_t tileSize) {
 
  SmallVector<Value, 4> tileSizes;
  for (unsigned i = 0; i < rank; ++i) {
    int64_t val = i == axis ? tileSize : 0;
    tileSizes.push_back(b.create<ConstantIndexOp>(loc, val));
  }
  return tileSizes;
}

//template <typename LoopTy>
Optional<TiledLinalgOp>
tileScopeImpl(
  OpBuilder& b, TileScope& ts, int64_t tileSize) {
  // early termination
  if (ts.tileOps.size() == 0) return llvm::None;

  // 1. Build the tiled loop ranges.
  //    Use lastop to create loops variables 
  auto lastOp = ts.tileOps.back().op;
  auto lastAxis = ts.tileOps.back().axis;
  auto lastRank = ts.tileOps.back().rank;
  auto loc = lastOp.getLoc();

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(lastOp);

  auto lastAllShapeSizes = lastOp.createFlatListOfOperandDims(b, loc);

  AffineMap lastShapeSizesToLoopsMap = lastOp.getShapesToLoopsMap();
  if (!lastShapeSizesToLoopsMap)
    return llvm::None;

  SmallVector<Range, 4> loopRanges = makeTiledLoopRange(
    b, loc, lastShapeSizesToLoopsMap, lastAllShapeSizes, lastAxis, tileSize);

  // iteratorTypes Attribute
  SmallVector<Attribute, 4> iteratorTypes;
  iteratorTypes.push_back(lastOp.iterator_types().getValue()[lastAxis]);
  
  // If interchangeVector is empty, use the identity. Build the permutation map
  // otherwise.
  auto invPermutationMap =
    AffineMap::getMultiDimIdentityMap(lastRank, b.getContext());
  // TODO support loop interchange later

  // 2. Create the tiled loops.
  SmallVector<Value, 8> lbsStorage, ubsStorage, stepsStorage;
  unpackRanges(loopRanges, lbsStorage, ubsStorage, stepsStorage);
  ValueRange lbs(lbsStorage), ubs(ubsStorage), steps(stepsStorage);
  SmallVector<Value, 8> ivs(lbs.size());

  // only first axis 
  auto tiledLoopBodyBuilder = 
    [&](OpBuilder& b, Location loc, ValueRange loopIvs) {
    ivs.assign(loopIvs.begin(), loopIvs.end());

    // When an `interchangeVector` is present, it has been applied to the
    // loop ranges and the iterator types. Apply its inverse to the
    // resulting loop `ivs` to match the op definition.
    SmallVector<Value, 4> interchangedIvs;

    // TODO support loop interchange later
    interchangedIvs.assign(ivs.begin(), ivs.end());

    // go through all ops
    for (auto& top : ts.tileOps) {
      auto linalgOp = top.op;
      auto axis = top.axis;
      auto rank = top.rank;

      auto localAllShapeSizes = linalgOp.createFlatListOfOperandDims(b, loc);
      AffineMap localShapeSizesToLoopsMap = linalgOp.getShapesToLoopsMap();

      auto localSizeBounds =
        applyMapToValues(b, loc, localShapeSizesToLoopsMap, localAllShapeSizes);

      // get all values for now
      // TODO: relax this later
      SmallVector<Value> valuesToTile = linalgOp.getInputAndOutputOperands();

      SmallVector<Value, 4> tileSizes =
        createTileSize(b, loc, axis, rank, tileSize);

      SmallVector<Value, 4> localTiledOperands = makeTiledShapes(
        b, loc, linalgOp, valuesToTile, interchangedIvs, tileSizes, localSizeBounds);

      SmallVector<Type, 4> resultTensorTypes;
      for (OpOperand* opOperand : linalgOp.getOutputTensorOperands())
        resultTensorTypes.push_back(
          localTiledOperands[opOperand->getOperandNumber()].getType());

      linalgOp.clone(b, loc, resultTensorTypes, localTiledOperands);
    }

  };

  auto loopNest = buildLoopNest(
    b, loc, 
    lbs.take_front(), ubs.take_front(), steps.take_front(),
    tiledLoopBodyBuilder);

  for (auto& top : ts.tileOps) {
    top.op.erase();
  }

  return None;
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
  //AffineMap invMap = inversePermutation(affineMap);
  AffineMap invMap = inverseAndBroadcastProjectedPermuation(affineMap);
  if(invMap.isEmpty()) return llvm::None;
  auto invComposed = invMap.compose(createOneHot(invMap.getNumInputs(), dimIndex));
  auto iterAxes = getAllIndicesForNonZeros(invComposed);
  // no support all-to-1 or non mapping
  if (iterAxes.size() != 1) {
    return llvm::None;
  }
  return iterAxes[0];
}

// update view to DimAndOps mapping
static void updateViewToDimAndOps(
  llvm::DenseMap<Value, DimProperty>& viewToDimProp,
  Value view, 
  LinalgOp op,
  AffineMap& affineMap, 
  unsigned iterAxis, 
  bool reduced, 
  bool assigned = false) {

  if (viewToDimProp.count(view) == 0) {
    auto composed = affineMap.compose(createOneHot(affineMap.getNumDims(), iterAxis));
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
  llvm::DenseMap<Value, DimProperty>& viewToDimAndOps,
  llvm::DenseMap<Operation*, std::pair<unsigned, unsigned>>& opToIterAxisAndRank,
  Operation* op) {
  if (!isa<LinalgOp>(op)) {
    return MemoryEffectOpInterface::hasNoEffect(op);
  }

  auto producer = cast<LinalgOp>(op);
  // only handle strucutre op
  if (!isStructuralLinalg(producer)) {
    return false;
  }

  unsigned iterAxis;
  int64_t numInputs = producer.getNumInputs();

  SmallVector<AffineMap> affineMaps
    = llvm::to_vector<4>(producer.indexing_maps().getAsValueRange<AffineMapAttr>());

  if (opToIterAxisAndRank.count(op) == 0) {
    auto producerOutView = producer.outputs()[0];
    if (viewToDimAndOps.count(producerOutView) == 0) {
      // no fuse non-producer-consumer op
      return false;
    }
    int outDimIndex = viewToDimAndOps[producerOutView].dim;
    if (outDimIndex == DimProperty::NoDim) return false;
    auto maybeIterAxis = getIterAxis(affineMaps[numInputs], outDimIndex);
    if (!maybeIterAxis.hasValue()) return false;
    iterAxis = maybeIterAxis.getValue();
    opToIterAxisAndRank[op] = { iterAxis, affineMaps[numInputs].getNumDims() };
  } else {
    iterAxis = opToIterAxisAndRank[op].first;
  }

  SmallVector<StringAttr> iterTypes 
    = llvm::to_vector<4>(producer.iterator_types().getAsRange<StringAttr>());
  // For now, avoid fusion along reduction iterator 
  if (isReductionIterator(iterTypes[iterAxis])) return false;

  auto iterOneHot = createOneHot(iterTypes.size(), iterAxis);

  // insert producer's input
  for (int64_t i = 0; i < numInputs; ++i) {
    updateViewToDimAndOps(viewToDimAndOps, producer.inputs()[i],
      producer, affineMaps[i], iterAxis, false/*performedReduce*/);
  }

  return true;
}

static bool isConsumerValidforTileScope(
  llvm::DenseMap<Value, DimProperty>& viewToDimProp,
  llvm::DenseMap<Operation*, std::pair<unsigned, unsigned>>& opToIterAxisAndRank,
  Operation* op) {
  if (!isa<LinalgOp>(op)) {
    return MemoryEffectOpInterface::hasNoEffect(op);
  }

  auto consumer = cast<LinalgOp>(op);

  // only handle strucutral op
  if (!isStructuralLinalg(consumer)) {
    return false;
  }

  SmallVector<AffineMap> affineMaps
    = llvm::to_vector<4>(consumer.indexing_maps().getAsValueRange<AffineMapAttr>());
  int64_t numInputs = consumer.getNumInputs();


  Optional<unsigned> maybeIterAxis = None;
  unsigned iterAxis;

  if (opToIterAxisAndRank.count(op) == 0) {

    bool atLeastOne = false;

    for (int64_t i = 0; i < numInputs; ++i) {
      auto input = consumer.inputs()[i];
      if (viewToDimProp.count(input) != 0) {
        int intDimIndex = viewToDimProp[input].dim;

        if (intDimIndex == DimProperty::NoDim) {
          return false;
        }
        auto inferredIterAxis = getIterAxis(affineMaps[i], intDimIndex);
        if (!inferredIterAxis.hasValue()) continue;
        if (!atLeastOne) {
          iterAxis = inferredIterAxis.getValue();
        } else if (iterAxis != inferredIterAxis.getValue()){
          return false;
        } 
        atLeastOne = true;
      } 
    }

    if (!atLeastOne) {
      return false;
    } 

    opToIterAxisAndRank[op] = { iterAxis, affineMaps[numInputs].getNumDims() };
  } else {
    iterAxis = opToIterAxisAndRank[op].first;
  }

  SmallVector<StringAttr> iterTypes
    = llvm::to_vector<4>(consumer.iterator_types().getAsRange<StringAttr>());

  bool performedReduced = isReductionIterator(iterTypes[iterAxis]);

  // insert consumer's output
  {
    updateViewToDimAndOps(viewToDimProp, consumer.outputs()[0],
      consumer, affineMaps[numInputs], iterAxis, performedReduced, true /*assigned*/);
  }

  // insert consumer' other input
  for (int64_t i = 0; i < numInputs; ++i) {
    updateViewToDimAndOps(viewToDimProp, consumer.inputs()[i],
      consumer, affineMaps[i], iterAxis, false/*performedReduced*/);
  }

  return true;
}

// Collect ops within the Block of a given AnchorOp
static void createTilingScope(
  TileScope& ts,
  unsigned iterAxis) {

  LinalgOp anchorOp = ts.anchorOp;
  Block* block = anchorOp->getBlock();

  // id to ops
  SmallVector<Operation*> ops;
  int anchorId = -1;
  // initialization
  for (auto& op : *block) {
    if (anchorId == -1 && anchorOp.getOperation() == &op) {
      anchorId = static_cast<int>(ops.size());
    }
    ops.push_back(&op);
  }
 
  SmallVector<AffineMap> affineMaps
    = llvm::to_vector<4>(anchorOp.indexing_maps().getAsValueRange<AffineMapAttr>());

  SmallVector<StringAttr> iterTypes
    = llvm::to_vector<4>(anchorOp.iterator_types().getAsRange<StringAttr>());

  // if iterAxis is out of bound, early return
  if (iterAxis >= iterTypes.size()) {
    return;
  }

  llvm::DenseMap<Value, DimProperty> viewToDimProp;
  llvm::DenseMap<Operation*, std::pair<unsigned, unsigned>> opToIterAxisAndRank;
 
  opToIterAxisAndRank[anchorOp] = { iterAxis, static_cast<unsigned>(affineMaps.back().getNumDims())};
  bool performedReduce = isReductionIterator(iterTypes[iterAxis]);

  // insert producer's input
  int64_t numInputs = anchorOp.getNumInputs();
  for (int64_t i = 0; i < numInputs; ++i) {
    updateViewToDimAndOps(viewToDimProp, anchorOp.inputs()[i],
      anchorOp, affineMaps[i], iterAxis, false/*performedReduce*/);
  }

  int64_t numOutputs = anchorOp.getNumOutputs();
  for (int64_t i = 0; i < numOutputs; ++i) {
    updateViewToDimAndOps(viewToDimProp, anchorOp.outputs()[i],
      anchorOp, affineMaps[numInputs+i], iterAxis, performedReduce, true/*assigned*/);
  }

  // use set here, instead of SmallSet, since we need ids to be sorted.
  std::set<int> determinedIds;
  determinedIds.insert(anchorId);

  int iterCnt = 0;
  while (iterCnt++ < static_cast<int>(block->getOperations().size())) {
    bool changed = false;
    //producer
    for (int i = anchorId - 1; i >= 0; --i) {
      if (determinedIds.count(i) > 0) continue;
      if (isProducerValidforTileScope(viewToDimProp, opToIterAxisAndRank, ops[i])) {
        changed = true;
        determinedIds.insert(i);
      } else {
        break;
      }
    }
    //consumer
    for (int i = anchorId + 1; i < static_cast<int>(ops.size()); ++i) {
      if (determinedIds.count(i) > 0) continue;
      if (isConsumerValidforTileScope(viewToDimProp, opToIterAxisAndRank, ops[i])) {
        changed = true;
        determinedIds.insert(i);
      } else {
        break;
      }
    }
    if (!changed) break;
  }
  
  for (auto id : determinedIds) {
    if (auto linalg = dyn_cast<LinalgOp>(ops[id])) {
      ts.tileOps.push_back({ops[id], opToIterAxisAndRank[ops[id]].first, 
                             opToIterAxisAndRank[ops[id]].second });
    }
  }
}
 
static void collectScopeFromAnchorOp(
  FuncOp func, StringRef anchorTag, 
  unsigned iterAxis, 
  SmallVectorImpl<TileScope>& collection) {

  SmallSet<Block*, 4> visitedBlocks;

  // collect op with anchorTag as intial values
  func.walk([&](LinalgOp op) {
    // skip non-targeting or visited block
    if (!op->hasAttr(anchorTag) || 
        visitedBlocks.contains(op->getBlock())) {
      return;
    }

    collection.emplace_back(op);
    visitedBlocks.insert(op->getBlock());
  });

  // create tiling scope for each anchorTag
  for (TileScope& ts : collection) {
    createTilingScope(ts, iterAxis);
  }
}

bool IsHoistUpOp(Operation* op) {
  return isa<memref::AllocOp>(op) ||
    isa<memref::CollapseShapeOp>(op) ||
    isa<memref::DimOp>(op) ||
    isa<memref::ExpandShapeOp>(op) ||
    isa<memref::ReshapeOp>(op);
}

bool IsHoistDownOp(Operation* op) {
  return isa<memref::DeallocOp>(op);
}

struct LinalgScopeTilingPass : public LinalgScopeTilingBase<LinalgScopeTilingPass> {
  LinalgScopeTilingPass() = default;
  LinalgScopeTilingPass(
    StringRef anchorTag, int64_t tileAxis,
    int64_t tileSize, LinalgTilingLoopType loopType,
    StringRef distributionType) {

    this->anchorTag.setValue(anchorTag.str());
    this->tileAxis = tileAxis;
    this->tileSize = tileSize;
    this->loopType = "";
    this->loopTypeEnum = loopType;
    this->distributionType = distributionType.str();
  }

  void runOnFunction() override {
    // early terminate when tileSize == 0
    if (tileSize == 0) return;

    // parse
    LinalgTilingLoopType type =
      llvm::StringSwitch<LinalgTilingLoopType>(loopType)
      .Case("for", LinalgTilingLoopType::Loops)
      .Case("affine", LinalgTilingLoopType::AffineLoops)
      .Case("parallel", LinalgTilingLoopType::ParallelLoops)
      .Case("tiled_loop", LinalgTilingLoopType::TiledLoops)
      .Default(loopTypeEnum);

    FuncOp funcOp = getFunction();

    auto& domInfo = getAnalysis<DominanceInfo>();
    auto& postDomInfo = getAnalysis<PostDominanceInfo>();

    // hoisting
    for (auto& block : funcOp.getBody()) {
      hoistUpOpsInBlock(&block, domInfo, IsHoistUpOp);
      hoistDownOpsInBlock(&block, postDomInfo, IsHoistDownOp);
    }

    SmallVector<TileScope> collection;
    collectScopeFromAnchorOp(funcOp, anchorTag, tileAxis, collection);

    OpBuilder b(funcOp.getContext());
    for (auto& ts : collection) {
      tileScopeImpl(b, ts, tileSize);
    }
  }

  LinalgTilingLoopType loopTypeEnum;
};

} // anonymous 

std::unique_ptr<FunctionPass>
mlir::createLinalgScopeTilingPass(
  StringRef anchorTag, int64_t tileAxis,
  int64_t tileSize, linalg::LinalgTilingLoopType loopType,
  StringRef distributionType) {
  return std::make_unique<LinalgScopeTilingPass>(anchorTag, tileAxis, tileSize, loopType,
    distributionType);
}
