//===- OpCnt.cpp ----------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Stat/OpCnt/OpCnt.h"
#include "byteir/Stat/Common/Reg.h"
#include "llvm/Support/CommandLine.h"

using namespace byteir;
using namespace mlir;

//===----------------------------------------------------------------------===//
// OpCnt registration
//===----------------------------------------------------------------------===//

void byteir::registerOpCntStatistics() {

  MLIRStatRegistration reg("op-cnt", [](ModuleOp module, raw_ostream &output) {
    return byteir::opCntStatistics(module, output);
  });
}

mlir::LogicalResult byteir::opCntStatistics(ModuleOp op,
                                            llvm::raw_ostream &os) {
  os << "========== Operation Type and Its Numbers ============\n";
  llvm::StringMap<unsigned> opCnt;

  for (FuncOp func : op.getOps<FuncOp>()) {
    func.walk([&](Operation *op) { opCnt[op->getName().getStringRef()] += 1; });
  }
  SmallVector<StringRef, 64> sorted(opCnt.keys());
  llvm::sort(sorted);
  for (auto opType : sorted) {
    os << opType << " " << opCnt[opType] << "\n";
  }
  return success();
}