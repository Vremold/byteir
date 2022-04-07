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
    return byteir::opCntStatistics(module, output,
                                   MLIRStatRegistration::fucnName,
                                   MLIRStatRegistration::topOnly);
  });
}

mlir::LogicalResult byteir::opCntStatistics(ModuleOp moduleOp,
                                            llvm::raw_ostream &os,
                                            const std::string &funcNmae,
                                            bool topOnly) {
  os << "========== Operation Type and Its Numbers ============\n";
  llvm::StringMap<unsigned> opCnt;

  if (funcNmae.empty()) {
    for (FuncOp func : moduleOp.getOps<FuncOp>()) {
      if (topOnly) {
        for (auto &op : func.getOps()) {
          opCnt[op.getName().getStringRef()] += 1;
        }
      } else {
        func.walk(
            [&](Operation *op) { opCnt[op->getName().getStringRef()] += 1; });
      }
    }
  } else {
    SymbolTable symbolTable(moduleOp);
    auto func = symbolTable.lookup<FuncOp>(funcNmae);

    // early return
    if (func == nullptr)
      return success();

    if (topOnly) {
      for (auto &op : func.getOps()) {
        opCnt[op.getName().getStringRef()] += 1;
      }
    } else {
      func.walk(
          [&](Operation *op) { opCnt[op->getName().getStringRef()] += 1; });
    }
  }

  SmallVector<StringRef, 64> sorted(opCnt.keys());
  llvm::sort(sorted);
  for (auto opType : sorted) {
    os << opType << " " << opCnt[opType] << "\n";
  }
  return success();
}