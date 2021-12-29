//===- Alias.h ------------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_ANALYSIS_ALIAS_H
#define BYTEIR_ANALYSIS_ALIAS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include <memory>

namespace byteir {

struct AliasAnalysis {
  AliasAnalysis(mlir::Block* b, 
                llvm::ArrayRef<mlir::Value> initial_copy,
                std::function<bool(mlir::Operation& op)> is_alias)
    : block(b), values(initial_copy.begin(), initial_copy.end()), is_alias(is_alias) {
    int cnt = values.size();
    for (int i = 0; i < cnt; ++i) {
      mlir::Value val = values[i];
      if (value_to_index.count(val) == 0) {
        value_to_index[val] = i;
        leader_to_index.insert(i);
      }
    }
  }

  virtual ~AliasAnalysis() {}

  virtual int GetOrCreateIndex(mlir::Value val) {
    if (value_to_index.count(val) == 0) {
      int count = values.size();
      value_to_index[val] = count;
      values.push_back(val);
      leader_to_index.insert(count);

    }
    return value_to_index[val];
  }

  // default RunOnBlock
  // check x = op(y)
  virtual void RunOnBlock() {
    if (block->empty()) return;

    for (auto& op : block->without_terminator()) {
      if (is_alias(op)) {
        int newId = GetOrCreateIndex(op.getResult(0));
        int oldId = GetOrCreateIndex(op.getOperand(0));
        if (newId <= oldId) {
          leader_to_index.unionSets(newId, oldId);
        } else {
          leader_to_index.unionSets(oldId, newId);
        }
      }
    }
  }

  int GetLeaderIndex(mlir::Value val) {
    return leader_to_index.getLeaderValue(value_to_index[val]);
  }

  mlir::Block* block; // a reference
  llvm::SmallVector<mlir::Value> values;
  std::function<bool(mlir::Operation& op)> is_alias;

  llvm::SmallDenseMap<mlir::Value, int> value_to_index;
  llvm::EquivalenceClasses<int> leader_to_index;
  
};



} // namespace byteir

#endif // BYTEIR_ANALYSIS_ALIAS_H