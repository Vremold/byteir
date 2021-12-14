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

// Count operation in a ModuleOp recursively.
mlir::LogicalResult opCntStatistics(mlir::ModuleOp op, llvm::raw_ostream &os);

} // namespace byteir

#endif // BYTEIR_STAT_OPCNT_OPCNT_H
