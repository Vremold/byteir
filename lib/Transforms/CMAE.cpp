//===- CMAE.cpp -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/CMAE.h"
#include "byteir/Utils/IRRewrite.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;

// CMAE use a conservative algorithm bying dominator and post dominator,
// without fine doing dependence analysis.
// For better results, maybe use mem2reg in llvm.
//
// The implemented algorihm in this CMAE:
//
// A load (L1) can be eliminated, if all the following satisfied.
// 1) its nearest dominator is a load (L2), and L1.indices == L2.indices,
//    L1 and L2 in the same block
// 2) (RAW checking) there is either
//   a) no side-effect op or store op (S), making L1 postdominate S.
//   b) if S, making L1 postdominate S, but there is another L3,
//      making L1 postdominate L3, L3 postdominate S, and L1.indices ==
//      L3.indices (basically checking L1 is not S's nearest load postdominator)
// A store (S1) can be eliminated, if all the following satisfied.
// 1) its nearest postDominator a store (S2), S1 and S1.indices == S2.indices
//    S1 and S2 in the same block
// 2) (RAW checking) there is either
//    a) no user or load (L), making S1 dominate L.
//    b) if L, making S1 dominate L, but there is another S3,
//       making S1 dominates S3, S3 dominates L, and S1.indices == S3.indices
//       (basically checking S1 is not L's nearest store dominator)

namespace {

struct CMAEPass : public CMAEBase<CMAEPass> {
  explicit CMAEPass(const std::string &skip) : CMAEBase() { skipAttr = skip; }
  void runOnOperation() final {
    auto f = getOperation();

    if (f->hasAttr(skipAttr)) {
      return;
    }

    runCMAEInFuncLike(f);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createCMAEPass(const std::string &skip) {
  return std::make_unique<CMAEPass>(skip);
};
