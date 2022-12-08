//===- AffineUtils.cpp ----------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/AffineUtils.h"
#include "byteir/Utils/Utils.h"

using namespace mlir;

/**
 * find iteration index through dim and inversePermutation
 * E.g. if affineMap = (d0, d1, d2)-> (d0, d2), dim = 1
 * Then invMap = (d0, d1)->(d0, 0, d1)
 *      oneHot = (0, 1)
 *      invComposed = (0, 0, 1)
 *      iterAxis = 2
 **/
FailureOr<unsigned> mlir::getIterAxisFromDim(AffineMap affineMap,
                                             unsigned dimIndex) {
  AffineMap invMap = inverseAndBroadcastProjectedPermutation(affineMap);
  if (invMap.isEmpty())
    return failure();
  auto invComposed =
      invMap.compose(createOneHot(invMap.getNumInputs(), dimIndex));
  auto iterAxes = getAllIndicesForNonZeros(invComposed);
  // no support all-to-1 or non mapping
  if (iterAxes.size() != 1) {
    return failure();
  }
  return iterAxes[0];
}
