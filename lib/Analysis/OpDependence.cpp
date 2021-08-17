//===- OpDependence.cpp ---------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Analysis/OpDependence.h"

#include "llvm/ADT/DenseMap.h"
#include <utility>  // pair


using namespace llvm;
using namespace mlir;


namespace mlir {
  struct OpDependenceInfoImpl {
    llvm::DenseMap<std::pair<Operation*, Operation*>, bool> memorized;
  };
}

namespace {
  bool properlyDependsRecursion(
    Operation* opFrom, Operation* opTo, 
    Block* block, llvm::DenseMap<std::pair<Operation*, Operation*>, bool>& memorized) {
    if (opFrom == nullptr || opTo == nullptr) return false;
    if (opFrom->getBlock() != block || opTo->getBlock() != block) return false;
    if (opFrom == opTo) return true;

    std::pair<Operation*, Operation*> p = { opFrom , opTo };
    auto found = memorized.find(p);

    if (found != memorized.end()) {
      return found->second;
    }

    // not found
    for (auto val : opTo->getOperands()) {
      if (properlyDependsRecursion(opFrom, val.getDefiningOp(), block, memorized)) {
        memorized[p] = true;
        return true;
      }
    }

    memorized[p] = false;
    return false;
  }
}

mlir::OpDependenceInfo::OpDependenceInfo(Block* b)
  : block_(b), impl_(new OpDependenceInfoImpl()) {}

mlir::OpDependenceInfo::~OpDependenceInfo() {}

// TODO: use a simpler algorithm by preprocessing block_
bool mlir::OpDependenceInfo::properlyDepends(Operation * opFrom, Operation * opTo) {
  if (opFrom == opTo) return false;
  return properlyDependsRecursion(opFrom, opTo, block_, impl_->memorized);
}

bool mlir::OpDependenceInfo::Depends(Operation* a, Operation* b) {
  return a == b || properlyDepends(a, b);
}

