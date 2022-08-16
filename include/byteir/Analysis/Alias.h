//===- Alias.h ------------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_ANALYSIS_ALIAS_H
#define BYTEIR_ANALYSIS_ALIAS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace byteir {

struct AliasAnalysis {
  AliasAnalysis(mlir::Block *b, llvm::ArrayRef<mlir::Value> initials,
                std::function<bool(mlir::Operation &op)> checkAlias)
      : block(b), values(initials.begin(), initials.end()),
        isAlias(checkAlias) {
    int cnt = values.size();
    for (int i = 0; i < cnt; ++i) {
      mlir::Value val = values[i];
      if (valueToIndex.count(val) == 0) {
        valueToIndex[val] = i;
        leaderToIndex.insert(i);
      }
    }
  }

  virtual ~AliasAnalysis() {}

  virtual int getOrCreateIndex(mlir::Value val) {
    if (valueToIndex.count(val) == 0) {
      int count = values.size();
      valueToIndex[val] = count;
      values.push_back(val);
      leaderToIndex.insert(count);
    }
    return valueToIndex[val];
  }

  // default runOnBlock
  // check x = op(y)
  virtual void runOnBlock() {
    if (block->empty())
      return;

    for (auto &op : block->without_terminator()) {
      if (isAlias(op)) {
        int newId = getOrCreateIndex(op.getResult(0));
        int newLeader = leaderToIndex.getLeaderValue(newId);
        int oldId = getOrCreateIndex(op.getOperand(0));
        int oldLeader = leaderToIndex.getLeaderValue(oldId);
        if (newLeader <= oldLeader) {
          leaderToIndex.unionSets(newLeader, oldLeader);
        } else {
          leaderToIndex.unionSets(oldLeader, newLeader);
        }
      }
    }
  }

  int getLeaderIndex(mlir::Value val) {
    return leaderToIndex.getLeaderValue(valueToIndex[val]);
  }

  mlir::Block *block; // a reference
  llvm::SmallVector<mlir::Value> values;
  std::function<bool(mlir::Operation &op)> isAlias;

  llvm::SmallDenseMap<mlir::Value, int> valueToIndex;
  llvm::EquivalenceClasses<int> leaderToIndex;
};

} // namespace byteir

#endif // BYTEIR_ANALYSIS_ALIAS_H