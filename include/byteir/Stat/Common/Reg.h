//===- Reg.h --------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_STAT_COMMON_REG_H
#define MLIR_STAT_COMMON_REG_H

#include "llvm/Support/CommandLine.h"
#include <string>

namespace llvm {
class StringRef;
class SourceMgr;
} // namespace llvm

namespace mlir {
class DialectRegistry;
struct LogicalResult;
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace mlir {

/// Interface of the function that do some statistics on the ModuleOp and
/// outputs the result to a stream.
using MLIRFunctionStat =
    std::function<LogicalResult(const ModuleOp &, llvm::raw_ostream &output)>;

using MLIRRegFunctionStat = std::function<LogicalResult(
    llvm::SourceMgr &sourceMgr, llvm::raw_ostream &output, MLIRContext *)>;

struct StatisticsParser : public llvm::cl::parser<const MLIRRegFunctionStat *> {
  StatisticsParser(llvm::cl::Option &opt);

  void printOptionInfo(const llvm::cl::Option &o,
                       size_t globalWidth) const override;
};

struct MLIRStatRegistration {

  static llvm::cl::opt<std::string> fucnName;

  static llvm::cl::opt<bool> topOnly;

  MLIRStatRegistration(llvm::StringRef name, const MLIRFunctionStat &function);
};
} // namespace mlir

#endif // MLIR_STAT_COMMON_REG_H