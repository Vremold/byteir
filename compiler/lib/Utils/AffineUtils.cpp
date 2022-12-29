//===- AffineUtils.cpp ----------------------------------------------------===//
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
  if (!affineMap.isProjectedPermutation())
    return failure();

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
