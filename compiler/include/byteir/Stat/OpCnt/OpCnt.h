//===- OpCnt.h  -------------------------------------------------*- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_STAT_OPCNT_OPCNT_H
#define BYTEIR_STAT_OPCNT_OPCNT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

namespace byteir {

void registerOpCntStatistics();

// Count operation within funcOps in a ModuleOp.
// funcName can be used to specify a specific function name
// If funcName is empty, all funcOps will be stat.
// When topOnly == true, only stat ops in a funcOps.
// If topOnly == false, stats will happen recursively
mlir::LogicalResult opCntStatistics(mlir::ModuleOp op, llvm::raw_ostream &os,
                                    const std::string &funcNmae = "",
                                    bool topOnly = false);

} // namespace byteir

#endif // BYTEIR_STAT_OPCNT_OPCNT_H
