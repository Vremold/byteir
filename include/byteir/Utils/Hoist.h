//===- Hoist.h ------------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_HOIST_H
#define BYTEIR_UTILS_HOIST_H

#include <functional>

namespace mlir {
class Value;
class Block;
class DominanceInfo;
class Operation;
class PostDominanceInfo;

// return least ProperlyDominant use or def
// Note: val must be one of refOp's operands
Operation* leastProperlyDominantUseOrDef(
  Value val,
  DominanceInfo& domInfo,
  Operation* refOp);

// return least ProperlyPostDominant use 
// Note: val must be one of refOp's operands or results
Operation* leastProperlyPostDominantUse(
  Value val,
  PostDominanceInfo& postDomInfo,
  Operation* refOp);

// return Operation Hoist Up within a Block of op
Operation* findHoistUpInBlock(
  Operation* op,
  DominanceInfo& domInfo);

// return Operation Hoist Down within a Block of op
Operation* findHoistDownInBlock(
  Operation* op,
  PostDominanceInfo& postDomInfo);

// hoist up ops in a given Block
void hoistUpOpsInBlock(
  Block* block, 
  DominanceInfo& domInfo,
  std::function<bool(Operation*)> checkFunc);

// hoist down ops in a given Block
void hoistDownOpsInBlock(
  Block* block,
  PostDominanceInfo& postDomInfo,
  std::function<bool(Operation*)> checkFunc);

} // namespace mlir

#endif // BYTEIR_UTILS_HOIST_H