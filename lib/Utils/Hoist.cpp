//===- Hoist.cpp --------------------------------------- -----------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/Hoist.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

// return least ProperlyDominant use or def
// Note: val must be refOp's operand
Operation* mlir::leastProperlyDominantUseOrDef(
  Value val, 
  DominanceInfo& domInfo, 
  Operation* refOp) {

  Operation* defOp = val.getDefiningOp();
  if (defOp == nullptr) return nullptr;
  Operation* curPos = defOp;
  for (Operation* user : val.getUsers()) {
    if (domInfo.properlyDominates(curPos, user) &&
        domInfo.properlyDominates(user, refOp)) {
      curPos = user;
    }
  }
  return curPos;
}

// return least ProperlyDominant use or def
Operation* mlir::leastProperlyPostDominantUse(
  Value val,
  PostDominanceInfo& postDomInfo,
  Operation* refOp) {

  Operation* curPos = nullptr;
  for (Operation* user : val.getUsers()) {
    bool isCurPosProperPostDominates =
      curPos != nullptr ? postDomInfo.properlyPostDominates(curPos, user)
                        : true;
    if (isCurPosProperPostDominates &&
        postDomInfo.properlyPostDominates(user, refOp)) {
      curPos = user;
    }
  }
  return curPos;
}

// return Operation Hoist Up within a Block of op
Operation* mlir::findHoistUpInBlock(
  Operation* op,
  DominanceInfo& domInfo) {
  Operation* curPos = &(op->getBlock()->front());
  for (auto val : op->getOperands()) {
    Operation* leastDominant 
      = leastProperlyDominantUseOrDef(val, domInfo, op);
    // skip nullptr
    if (leastDominant == nullptr) continue;

    if (domInfo.properlyDominates(curPos, leastDominant)) {
      curPos = leastDominant;
    }
  }

  return curPos;
}

Operation* mlir::findHoistDownInBlock(
  Operation* op,
  PostDominanceInfo& postDomInfo) {
  Operation* curPos = op->getBlock()->getTerminator();

  // check all results
  for (auto val : op->getResults()) {
    Operation* leastPostDominant
      = leastProperlyPostDominantUse(val, postDomInfo, op);
    if (leastPostDominant == nullptr) continue;
    if (postDomInfo.properlyPostDominates(curPos, leastPostDominant)) {
      curPos = leastPostDominant;
    }
  }

  for (auto val : op->getOperands()) {
    Operation* leastPostDominant
      = leastProperlyPostDominantUse(val, postDomInfo, op);
    if (leastPostDominant == nullptr) continue;
    if (postDomInfo.properlyPostDominates(curPos, leastPostDominant)) {
      curPos = leastPostDominant;
    }
  }

  return curPos;
}

// hoist up ops in a given Block
void mlir::hoistUpOpsInBlock(
  Block* block,
  DominanceInfo& domInfo,
  std::function<bool(Operation*)> checkFunc) {
  // early termination
  if (block == nullptr) return;

  SmallVector<std::pair<Operation*, Operation*>> moveAfterOps;

  // hanlde HoistUp 
  for (auto& op : block->without_terminator()) {
    // skip non-hoistable op
    if (!checkFunc(&op)) continue;

    auto pos = findHoistUpInBlock(&op, domInfo);
    if (pos != &op) {
      moveAfterOps.emplace_back(&op, pos);
    }
  }

  for (auto& p : moveAfterOps) {
    p.first->moveAfter(p.second);
  }
}

// hoist down ops in a given Block
void mlir::hoistDownOpsInBlock(
  Block* block,
  PostDominanceInfo& postDomInfo,
  std::function<bool(Operation*)> checkFunc) {
  // early termination
  if (block == nullptr) return;

  SmallVector<std::pair<Operation*, Operation*>> moveBeforeOps;

  // hanlde HoistUp 
  for (auto it = block->rbegin(); it != block->rend(); ++it) {
    auto& op = *it;
    // skip non-hoistable op
    if (!checkFunc(&op)) continue;
    auto pos = findHoistDownInBlock(&op, postDomInfo);
    if (pos != &op) {
      moveBeforeOps.emplace_back(&op, pos);
    }
  }

  for (auto& p : moveBeforeOps) {
    p.first->moveBefore(p.second);
  }
}
