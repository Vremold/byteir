//===- Transforms.cpp -----------------------------------------*--- C++ -*-===//
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
// Some code from Tiling.cpp of IREE
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Some code from TileUsingInterface.cpp of LLVM project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Utils/Transforms.h"

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Utils/AffineUtils.h"
#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalg_ext;
using namespace mlir::scf;

using IteratorTypes = llvm::SmallVector<llvm::Optional<utils::IteratorType>>;

//===----------------------------------------------------------------------===//
// mergeLoopIteratorTypes
//===----------------------------------------------------------------------===//

void mlir::linalg_ext::mergeLoopIteratorTypes(
    llvm::SmallVector<llvm::Optional<utils::IteratorType>> &from,
    llvm::SmallVector<llvm::Optional<utils::IteratorType>> &to) {
  // logic:
  // parallel, parallel => parallel
  // parallel, none => parallel
  // parallel, reduce => reduce
  // none, none => none
  // none, reduce => reduce
  // reduce, x => reduce
  for (auto &en : llvm::enumerate(from)) {
    if (en.value().has_value()) {
      if (to[en.index()].has_value() &&
          en.value().value() != to[en.index()].value()) {
        // when (iterTy, curTy) == (parallel, reduce) or (reduce, parallel)
        // assign iterTy = reduce
        to[en.index()] = utils::IteratorType::reduction;
      } else {
        // when either iterTy is none or iterTy == curTy
        // assign iterTy = curTy
        to[en.index()] = en.value().value();
      }
    }
  } // for en : llvm::enumerate(from)
}

//===----------------------------------------------------------------------===//
// labletateTileLoopType
//===----------------------------------------------------------------------===//

void mlir::scf::labelTileLoopType(Operation *op, ArrayRef<scf::ForOp> loops) {
  if (op == nullptr) {
    return;
  }

  auto innerMostSCFFor = loops.back();
  if (innerMostSCFFor.getBody() != op->getBlock()) {
    return;
  }

  IteratorTypes iterTys(loops.size(), llvm::None);

  for (auto &innerOp : innerMostSCFFor.getBody()->without_terminator()) {
    if (!isa<TilingInterface>(innerOp)) {
      continue;
    }

    FailureOr<IteratorTypes> curTys = getLoopIteratorTypes(&innerOp, loops);
    if (failed(curTys)) {
      continue;
    }

    // merge IteratorTypes to iterTys
    mergeLoopIteratorTypes(curTys.value(), iterTys);
  }

  auto ctx = op->getContext();
  for (auto &en : llvm::enumerate(iterTys)) {
    if (en.value().has_value() && en.value() == utils::IteratorType::parallel) {
      loops[en.index()]->setAttr(getSCFForParallelAttrName(),
                                 UnitAttr::get(ctx));
    }
  }
}

//===----------------------------------------------------------------------===//
// isValidTiling
//===----------------------------------------------------------------------===//

LogicalResult mlir::scf::isValidTiling(Operation *tiled) {
  if (tiled == nullptr)
    return failure();

  if (auto linalgExtOp = dyn_cast<linalg_ext::LinalgExtOp>(tiled)) {
    return linalgExtOp.isValidTiling(tiled);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// getLoopIteratorTypes
//===----------------------------------------------------------------------===//

namespace {

// return getIndexingMapsArray if an op having getIndexingMapsArray
FailureOr<llvm::SmallVector<AffineMap>> getIndexingMapsArray(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    return linalgOp.getIndexingMapsArray();
  } else if (auto linalgExtOp = dyn_cast<linalg_ext::LinalgExtOp>(op)) {
    return linalgExtOp.getIndexingMapsArray();
  }
  return failure();
}

bool isNewValue(Value val) {
  if (auto def = val.getDefiningOp()) {
    return isa<tensor::EmptyOp>(def);
  }
  return false;
}

FailureOr<bool> getLocalComputation(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    return llvm::all_of(linalgOp.getDpsInitOperands(), [&](OpOperand *opVal) {
      return isNewValue(opVal->get());
    });
  } else if (auto linalgExtOp = dyn_cast<linalg_ext::LinalgExtOp>(op)) {
    return llvm::all_of(linalgExtOp.getOutputOperands(), [&](OpOperand *opVal) {
      return isNewValue(opVal->get());
    });
  }
  return failure();
}

} // namespace

FailureOr<IteratorTypes>
mlir::linalg_ext::getLoopIteratorTypes(Operation *op,
                                       ArrayRef<scf::ForOp> loops) {
  // early termination if no TilingInterface
  if (!isa<TilingInterface>(op)) {
    return failure();
  }

  FailureOr<llvm::SmallVector<AffineMap>> indexingMaps =
      getIndexingMapsArray(op);
  // early termination if no indexingMaps
  // TODO: relax this, by making indexingMaps all reduce
  if (failed(indexingMaps)) {
    return failure();
  }

  FailureOr<bool> localComputation = getLocalComputation(op);
  // early termination if no indexingMaps
  // TODO: relax this, by making localComputation false.
  if (failed(localComputation)) {
    return failure();
  }

  auto tilingLoopIterType = cast<TilingInterface>(op).getLoopIteratorTypes();
  IteratorTypes retIterTys(loops.size(), llvm::None);

  // preset LoopIV to loopIdx
  DenseMap<Value, size_t> loopIV2Idx;
  for (const auto &en : llvm::enumerate(loops)) {
    auto forOp = en.value();
    loopIV2Idx[forOp.getInductionVar()] = en.index();
  }

  // check all args
  for (const auto &en : llvm::enumerate(op->getOperands())) {
    llvm::SmallVector<::mlir::OpFoldResult, 4> mixedOffsets;

    if (auto sliceOp = en.value().getDefiningOp<tensor::ExtractSliceOp>()) {
      mixedOffsets = sliceOp.getMixedOffsets();
    } else if (auto subviewOp = en.value().getDefiningOp<memref::SubViewOp>()) {
      mixedOffsets = subviewOp.getMixedOffsets();
    } else {
      continue;
    }

    auto indexingMap = indexingMaps.value()[en.index()];
    for (const auto &en2 : llvm::enumerate(mixedOffsets)) {
      Value argVal = en2.value().dyn_cast<Value>();

      if (!argVal || loopIV2Idx.count(argVal) == 0) {
        // skip when argVal folded to a const or not in loopIV2Idx
        // implying not a loop iv
        continue;
      }

      FailureOr<unsigned> iterAxis =
          getIterAxisFromDim(indexingMap, en2.index());
      if (failed(iterAxis)) {
        // skip when iterAxis not found
        continue;
      }

      auto iterTy = localComputation.value()
                        ? utils::IteratorType::parallel
                        : tilingLoopIterType[iterAxis.value()];
      auto loopIdx = loopIV2Idx[argVal];
      if (retIterTys[loopIdx].has_value()) {
        if (retIterTys[loopIdx].value() != iterTy) {
          // detect more than one LoopIterType
          return failure();
        }
      } else {
        // if has no value, set it now
        retIterTys[loopIdx] = iterTy;
      }
    } // for en2 : llvm::enumerate(mixedOffsets)
  }   // for en : llvm::enumerate(op->getOperands()))

  return retIterTys;
}

//===----------------------------------------------------------------------===//
// tileConsumerAndFuseProducerUsingSCFForOpExt
//===----------------------------------------------------------------------===//

namespace {

/// Return the untiled producer whose slice is used in a tiled consumer. The
/// method traverses the tile loop nest (`loops`) if needed, and returns the
/// `iter_args` of the outer most that is encountered. Traversing the iter_args
/// indicates that this is a destination operand of the consumer. If there was
/// no loop traversal needed, the second value of the returned tuple is empty.
static std::tuple<OpResult, Optional<OpOperand *>>
getUntiledProducerFromSliceSource(OpOperand *source,
                                  ArrayRef<scf::ForOp> loops) {
  Optional<OpOperand *> destinationIterArg;
  auto loopIt = loops.rbegin();
  while (auto iterArg = source->get().dyn_cast<BlockArgument>()) {
    scf::ForOp loop = *loopIt;
    if (iterArg.getOwner()->getParentOp() != loop)
      break;
    source = &loop.getOpOperandForRegionIterArg(iterArg);
    loopIt++;
  }
  if (loopIt == loops.rend()) {
    destinationIterArg = source;
  }
  return {source->get().dyn_cast<OpResult>(), destinationIterArg};
}

/// If the tiled operation is destination passing style, update the
/// slice of the destination used (which refers to the untiled destination)
/// to use the corresponding region argument of the innermost loop.
///
/// ```mlir
/// %0 =
/// scf.for %iv0 = ... iter_args(%arg = %0) {
///   %1 = tensor.extract_slice %0
///   %2 = tiled_op
///   %3 = tensor.insert_slice %2 into %arg
///   scf.yield %3
/// }
/// ```
///
/// is transformed to
///
/// ```mlir
/// scf.for %iv0 = ... iter_args(%arg = %0) {
///   %1 = tensor.extract_slice %arg
///   %2 = tiled_op
///   %3 = tensor.insert_slice %2 into %arg
///   scf.yield %3
/// }
/// ```
static void
updateDestinationOperandsForTiledOp(OpBuilder &builder,
                                    ValueRange tiledOpDestinationValues,
                                    ValueRange bbArgsList) {
  for (const auto &destValue : llvm::enumerate(tiledOpDestinationValues)) {
    auto sliceOp = destValue.value().getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp)
      continue;
    sliceOp.setOperand(0, bbArgsList[destValue.index()]);
  }
}

// update replacements when oldLoops changing to newLoops
static void updateReplacements(llvm::DenseMap<Value, Value> &replacements,
                               ArrayRef<scf::ForOp> oldLoops,
                               ArrayRef<scf::ForOp> newLoops) {
  // generate loop map
  llvm::DenseMap<scf::ForOp, scf::ForOp> oldToNewLoop;
  for (const auto &en : llvm::enumerate(oldLoops)) {
    oldToNewLoop[en.value()] = newLoops[en.index()];
  }

  for (auto &it : replacements) {
    if (auto oldResult = dyn_cast<OpResult>(it.second)) {
      if (auto oldLoop = dyn_cast<scf::ForOp>(oldResult.getOwner())) {
        if (oldToNewLoop.count(oldLoop) > 0) {
          auto newResult =
              oldToNewLoop[oldLoop]->getResult(oldResult.getResultNumber());
          it.second = newResult;
        }
      }
    }
  }
}

/// For a value to be yielded (`yieldedValue`) from within a loop nest `loops`,
/// construct the destructive update pattern that inserts the yielded
/// value into a destination tensor provided by `initValue` at offset
/// `tileOffsets` and size `tileSizes`. For example,
///
/// ```mlir
/// scf.for %iv0 = ... {
///   %0 = tiled_op
/// }
/// ```
///
/// is transformed to
///
/// ```mlir
/// scf.for %iv0 = ... iter_args(%arg = %0) {
///   %1 = tensor.extract_slice %arg
///   %2 = tiled_op
///   %3 = tensor.insert_slice %2 into %arg
///   scf.yield %3
/// }
/// ```
/// TODO: This API can be cleaned up by using `SubsetExtractOpInterface`.
///
/// This function is modified by adding functionality of updating replacements
static FailureOr<SmallVector<Value>>
yieldTiledValues(RewriterBase &rewriter, ValueRange initValues,
                 ValueRange yieldedValues,
                 ArrayRef<SmallVector<OpFoldResult>> tileOffsetsList,
                 ArrayRef<SmallVector<OpFoldResult>> tileSizesList,
                 MutableArrayRef<scf::ForOp> loops,
                 llvm::DenseMap<Value, Value> &replacements) {
  NewYieldValueFn yieldValueFn =
      [&](OpBuilder &b, Location loc,
          ArrayRef<BlockArgument> newBBArgs) -> SmallVector<Value> {
    SmallVector<Value> inserts;
    for (const auto &yieldedValue : llvm::enumerate(yieldedValues)) {
      ArrayRef<OpFoldResult> tileOffsets =
          tileOffsetsList[yieldedValue.index()];
      ArrayRef<OpFoldResult> tileSizes = tileSizesList[yieldedValue.index()];
      SmallVector<OpFoldResult> tileStrides(tileOffsets.size(),
                                            b.getIndexAttr(1));
      Value insert = b.create<tensor::InsertSliceOp>(
          loc, yieldedValue.value(), newBBArgs[yieldedValue.index()],
          tileOffsets, tileSizes, tileStrides);
      inserts.push_back(insert);
    }
    return inserts;
  };

  SmallVector<scf::ForOp> newLoops =
      replaceLoopNestWithNewYields(rewriter, loops, initValues, yieldValueFn,
                                   /*replaceIterOperandsUsesInLoop =*/false);

  // this functionality is added on top of the exisitng upstream version
  updateReplacements(replacements, loops, newLoops);

  for (const auto &loop : llvm::enumerate(loops)) {
    rewriter.eraseOp(loop.value());
    loops[loop.index()] = newLoops[loop.index()];
  }
  return llvm::to_vector(llvm::map_range(
      loops.front().getResults().take_back(yieldedValues.size()),
      [](OpResult r) -> Value { return r; }));
}

// create insertSliceOp for results
static LogicalResult
createResultSlices(RewriterBase &rewriter, Operation *op, Operation *tiledOp,
                   tensor::ExtractSliceOp sliceOp,
                   SmallVector<scf::ForOp> &loops,
                   llvm::DenseMap<Value, Value> &replacements) {
  if (!isa<TilingInterface>(op)) {
    return failure();
  }

  SmallVector<Value> destinationTensors; // tensor before tiling.
  if (failed(tensor::getOrCreateDestinations(rewriter, op->getLoc(), op,
                                             destinationTensors))) {
    return rewriter.notifyMatchFailure(op, "failed to get destinations");
  }

  int64_t numResults = op->getNumResults();
  SmallVector<SmallVector<OpFoldResult>> resultOffsetsList(numResults),
      resultSizesList(numResults);
  auto outputRange = tiledOp->getOperands().take_back(numResults);
  for (const auto &result : llvm::enumerate(op->getResults())) {
    auto tiledOutput = outputRange[result.index()];
    if (auto sliceOp =
            dyn_cast<tensor::ExtractSliceOp>(tiledOutput.getDefiningOp())) {
      resultOffsetsList[result.index()] = sliceOp.getMixedOffsets();
      resultSizesList[result.index()] = sliceOp.getMixedSizes();
    } else {
      // TODO: handle non-slice by creating a entire view
      return failure();
    }
  }

  auto oldNumResult = loops.front()->getNumResults();

  FailureOr<SmallVector<Value>> replacementOr =
      yieldTiledValues(rewriter, destinationTensors, tiledOp->getResults(),
                       resultOffsetsList, resultSizesList, loops, replacements);

  if (failed(replacementOr)) {
    return rewriter.notifyMatchFailure(op, "failed to yield replacement");
  }

  if (auto dstOp = dyn_cast<DestinationStyleOpInterface>(tiledOp)) {
    auto innerMostLoop = loops.back();
    SmallVector<Value> destinationTensors = dstOp.getDpsInitOperands();

    updateDestinationOperandsForTiledOp(
        rewriter, destinationTensors,
        innerMostLoop.getRegionIterArgs().take_back(destinationTensors.size()));
  }

  // update replacements
  for (const auto &en : llvm::enumerate(op->getResults())) {
    replacements[en.value()] =
        loops.front()->getResult(oldNumResult + en.index());
  }

  return success();
}

static void getProducerAndConsumerTensorSlices(
    Operation *op, llvm::DenseMap<Value, Value> &iterArgToOperand,
    SmallPtrSetImpl<Operation *> &opCollection,
    SmallPtrSetImpl<Value> &valCollection) {
  for (const auto val : op->getOperands()) {
    if (auto sliceOp = val.getDefiningOp<tensor::ExtractSliceOp>()) {
      // insert to opCollection
      if (!opCollection.contains(sliceOp))
        opCollection.insert(sliceOp);

      Value src = sliceOp.getSource();
      if (iterArgToOperand.count(src) > 0 &&
          !valCollection.contains(iterArgToOperand[src])) {
        valCollection.insert(iterArgToOperand[src]);
      }
    }
  }

  for (const auto val : op->getResults()) {
    for (const auto userOp : val.getUsers()) {
      if (auto sliceOp = dyn_cast<tensor::InsertSliceOp>(userOp)) {
        if (!opCollection.contains(userOp))
          opCollection.insert(userOp);
        Value dst = sliceOp.getDest();
        if (iterArgToOperand.count(dst) > 0 &&
            !valCollection.contains(iterArgToOperand[dst])) {
          valCollection.insert(iterArgToOperand[dst]);
        }
      }
    }
  }
}

} // namespace

FailureOr<scf::SCFTileAndFuseResult>
mlir::scf::tileConsumerAndFuseProducerUsingSCFForOpExt(
    RewriterBase &rewriter, TilingInterface consumer,
    const scf::SCFTileAndFuseOptions &options) {
  // This transformation is only valid for ops that return values (i.e. not
  // valid to use with operations that have memref operands).
  if (!consumer->getNumResults()) {
    return rewriter.notifyMatchFailure(
        consumer, "invalid pattern for op with no results");
  }

  // 1. First tile the consumer.
  scf::SCFTileAndFuseResult tileAndFuseResult;
  llvm::SmallDenseMap<Value, int64_t> yieldedValueToResultNumber;
  {
    FailureOr<scf::SCFTilingResult> tilingResult =
        tileUsingSCFForOp(rewriter, consumer, options.tilingOptions);
    if (failed(tilingResult))
      return rewriter.notifyMatchFailure(consumer, "failed to tile consumer");

    if (failed(isValidTiling(tilingResult->tiledOp))) {
      return rewriter.notifyMatchFailure(
          consumer, "failed to tile consumer due to invalid tiling");
    }

    tileAndFuseResult.tiledAndFusedOps.insert(tilingResult->tiledOp);
    tileAndFuseResult.loops = std::move(tilingResult->loops);
    for (const auto &result : llvm::enumerate(
             llvm::zip(consumer->getResults(), tilingResult->replacements))) {
      tileAndFuseResult.replacements[std::get<0>(result.value())] =
          std::get<1>(result.value());
      yieldedValueToResultNumber[tilingResult->tiledOp->getResult(
          result.index())] = result.index();
    }
  }

  // If there are no loops generated, fusion is immaterial.
  if (tileAndFuseResult.loops.empty())
    return tileAndFuseResult;

  // 2. Typically, the operands of the tiled operation are slices of the
  //    operands of the untiled operation. These are expressed in IR using
  //    `tensor.extract_slice` operations with source being the operands of the
  //    untiled operation. Create a worklist of these `tensor.extract_slice`
  //    operations. If the producers of the source of the `tensor.extract_slice`
  //    can be tiled such that the tiled value is generated in-place, that
  //    effectively tiles + fuses the operations.
  auto addCandidateSlices = [](Operation *fusedOp,
                               std::deque<tensor::ExtractSliceOp> &candidates) {
    for (Value operand : fusedOp->getOperands())
      if (auto sliceOp = operand.getDefiningOp<tensor::ExtractSliceOp>())
        candidates.push_back(sliceOp);
  };

  std::deque<tensor::ExtractSliceOp> candidates;
  addCandidateSlices(tileAndFuseResult.tiledAndFusedOps.back(), candidates);
  OpBuilder::InsertionGuard g(rewriter);
  while (!candidates.empty()) {

    // 2a. Traverse the slices in BFS fashion.
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();

    candidates.pop_front();

    // 2b. Get the producer of the source (potentially walking through
    // `iter_args` of nested `scf.for`)
    auto [fusableProducer, destinationIterArg] =
        getUntiledProducerFromSliceSource(&candidateSliceOp->getOpOperand(0),
                                          tileAndFuseResult.loops);
    if (!fusableProducer) {
      continue;
    }

    // 2c. Generate the tiled implementation of the producer of the source
    rewriter.setInsertionPoint(candidateSliceOp);

    FailureOr<Value> fusedProducerValue =
        tensor::replaceExtractSliceWithTiledProducer(rewriter, candidateSliceOp,
                                                     fusableProducer);

    if (failed(fusedProducerValue)) {
      continue;
    }

    Operation *unFusedProducerOp = fusableProducer.getOwner();
    Operation *fusedProducerOp = fusedProducerValue->getDefiningOp();
    if (auto linalgExtUnfused = dyn_cast<LinalgExtOp>(unFusedProducerOp)) {
      if (failed(linalgExtUnfused.correctTiledConsumerOps(
              rewriter, fusedProducerOp, fusableProducer.getResultNumber()))) {
        return failure();
      }
    }

    rewriter.replaceOp(candidateSliceOp, fusedProducerValue.value());

    if (failed(createResultSlices(rewriter, fusableProducer.getOwner(),
                                  fusedProducerOp, candidateSliceOp,
                                  tileAndFuseResult.loops,
                                  tileAndFuseResult.replacements))) {
      continue;
    }

    // put fused one in tileAndFuseResult
    if (!tileAndFuseResult.fusedProducers.contains(fusableProducer.getOwner()))
      tileAndFuseResult.fusedProducers.insert(fusableProducer.getOwner());

    // 2d. The operands of the fused producer might themselved be slices of
    //     values produced by operations that implement the `TilingInterface`.
    //     Add these operations to the worklist.
    tileAndFuseResult.tiledAndFusedOps.insert(fusedProducerOp);
    addCandidateSlices(fusedProducerOp, candidates);

    // 2e. If the slice is for a destination operand, for example,
    //
    // ```mlir
    // %0 = linalg.init
    // %1 = linalg.fill .. outs(%0 : )
    // %2 = scf.for .. iter_args(%arg0 = %1) {
    //   %3 = scf.for .. iter_args(%arg1 = %arg0) {
    //     %4 = tensor.extract_slice %arg1 [..]
    //     .. = linalg.matmul .. outs(%4 : )
    //   }
    // }
    // ```
    //
    // the IR is currently
    //
    // ```
    // %0 = linalg.init
    // %1 = linalg.fill
    // %2 = scf.for .. iter_args(%arg0 = %1 /* incorrect value */ ) {
    //   %3 = scf.for .. iter_args(%arg1 = %arg0) {
    //     %4 = tensor.extract_slice %0 /*incorrect value */ [..]
    //     %5 = linalg.fill .. outs(%4 : )
    //     .. = linalg.matmul .. outs(%5 : )
    //   }
    // }
    // ```
    //
    // The untiled `linalg.fill` is still used as the `init_value` since it
    // was originally a destination operand of the untiled `linalg.matmul`.
    // When fusing an operand that is a destination operand.
    //   - Update the iter_arg of the outer most loop to use the destination
    //     of the untiled producer.
    //   - Update the destination of the slice of the tiled producer generated
    //     to use the same basic block argument as the slice that was used to
    //     generate inplace the tiled implementation of the producer.
    // With this the IR will be.
    //
    // ```
    // %0 = linalg.init
    // %1 = scf.for .. iter_args(%arg0 = %0 /* corrected value */ ) {
    //   %2 = scf.for .. iter_args(%arg1 = %arg0) {
    //     %3 = tensor.extract_slice %arg1 /* corrected value */ [..]
    //     %4 = linalg.fill .. outs(%3 : )
    //     .. = linalg.matmul .. outs(%4 : )
    //   }
    // }
    // ```
    // TODO: This can be modeled better if the `DestinationStyleOpInterface`.
    // Update to use that when it does become available.
    scf::ForOp outerMostLoop = tileAndFuseResult.loops.front();
    Optional<unsigned> iterArgNumber;
    if (destinationIterArg) {
      iterArgNumber = outerMostLoop.getIterArgNumberForOpOperand(
          *destinationIterArg.value());
    }
    if (iterArgNumber) {
      int64_t resultNumber = fusableProducer.getResultNumber();
      if (auto dstOp = dyn_cast<DestinationStyleOpInterface>(
              fusableProducer.getOwner())) {
        outerMostLoop.setIterArg(
            iterArgNumber.value(),
            dstOp.getTiedOpOperand(fusableProducer)->get());
      }

      if (auto dstOp = fusedProducerValue
                           ->getDefiningOp<DestinationStyleOpInterface>()) {
        scf::ForOp innerMostLoop = tileAndFuseResult.loops.back();
        updateDestinationOperandsForTiledOp(
            rewriter, dstOp.getDpsInitOperand(resultNumber)->get(),
            innerMostLoop.getRegionIterArgs()[iterArgNumber.value()]);
      }
    }
  }

  // 3. clean loops args and unused loop carries

  // collect all iterArg2Operand for quick access later
  // iterArg2Operand as mapping from Loop's RegionIterArgs to IterOperands
  llvm::DenseMap<Value, Value> iterArg2Operand;
  for (auto &forOp : tileAndFuseResult.loops) {
    for (auto it : llvm::zip(forOp.getRegionIterArgs(), // iter inside region
                             forOp.getIterOperands()    // iter from outside
                             )) {
      iterArg2Operand.try_emplace(std::get<0>(it), std::get<1>(it));
    }
  }

  // check getLoopIteratorTypes for each fusedOp
  // if parallel, corresponding getRegionIterArgs will be simplified
  for (auto fusedOp : tileAndFuseResult.tiledAndFusedOps) {
    auto loopIterTypes = getLoopIteratorTypes(fusedOp, tileAndFuseResult.loops);
    if (failed(loopIterTypes)) {
      continue;
    }

    // collect all involed ops with the fusedOp
    SmallPtrSet<Operation *, 8> opCollection;
    SmallPtrSet<Value, 16> valCollection;
    opCollection.insert(fusedOp);
    // get producer and consumer slices
    getProducerAndConsumerTensorSlices(fusedOp, iterArg2Operand, opCollection,
                                       valCollection);

    for (const auto &en : llvm::enumerate(loopIterTypes.value())) {
      if (en.value().has_value() &&
          en.value().value() == utils::IteratorType::parallel) {
        auto forOp = tileAndFuseResult.loops[en.index()];
        for (auto it :
             llvm::zip(forOp.getRegionIterArgs(), // iter inside region
                       forOp.getIterOperands()    // iter from outside
                       )) {
          // only replace involved value, they are
          // 1) producer or consumer slices with the given fusedOp
          // 2) all involved loops' getIterOperands, which handle nested cases
          std::get<0>(it).replaceUsesWithIf(
              std::get<1>(it), [&](OpOperand &use) {
                return opCollection.contains(use.getOwner()) ||
                       valCollection.contains(use.get());
              });
        }
      }
    }
  }

  return tileAndFuseResult;
}

namespace mlir {
namespace linalg_ext {
// Marker used as attribute name in generated Linalg rewriting transformations.
const StringLiteral LinalgTransforms::kLinalgTransformMarker =
    "__internal_linalg_transform__";

LinalgTransformationFilter::LinalgTransformationFilter(
    ArrayRef<StringAttr> matchDisjunction, Optional<StringAttr> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {}

LinalgTransformationFilter::LinalgTransformationFilter(
    const FilterFunction &f, ArrayRef<StringAttr> matchDisjunction,
    Optional<StringAttr> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {
  if (f)
    filters.push_back(f);
}

LogicalResult
LinalgTransformationFilter::checkAndNotify(PatternRewriter &rewriter,
                                           Operation *op) const {
  if (llvm::any_of(filters,
                   [&](const FilterFunction &f) { return failed(f(op)); }))
    return failure();

  auto attr = op->template getAttrOfType<StringAttr>(
      LinalgTransforms::kLinalgTransformMarker);

  if (!attr) {
    // 1. Has no filter case and matchDisjunction is empty.
    if (matchDisjunction.empty() || matchByDefault)
      return success();

    // 2. Has no filter but was expecting a filter.
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << " does not have any filter from list: ";
      interleaveComma(matchDisjunction, diag);
    });
  }

  // 4. Match explicit filter.
  for (auto filter : matchDisjunction)
    if (attr.getValue() == filter)
      return success();

  // 5. Fail to match.
  return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
    diag << " does not have any filter from list: ";
    interleaveComma(matchDisjunction, diag);
  });
}

void LinalgTransformationFilter::replaceLinalgTransformationFilter(
    PatternRewriter &rewriter, Operation *op) const {
  if (replacement.has_value())
    op->setAttr(LinalgTransforms::kLinalgTransformMarker, replacement.value());
  else
    op->removeAttr(
        rewriter.getStringAttr(LinalgTransforms::kLinalgTransformMarker));
}

bool LinalgTransformationFilter::hasReplacementFilter(Operation *op) const {
  if (!replacement)
    return false;
  auto attr = op->getAttr(LinalgTransforms::kLinalgTransformMarker)
                  .dyn_cast<StringAttr>();
  return attr && attr == *replacement;
}
} // namespace linalg_ext
} // namespace mlir
