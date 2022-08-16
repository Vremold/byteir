//===- SideEffect.h -------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_ANALYSIS_SIDEEFFECT_H
#define BYTEIR_ANALYSIS_SIDEEFFECT_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <string>

namespace mlir {
class Operation;
}

namespace byteir {

// utils for io side effect
enum ArgSideEffectType : int {
  kInput = 0, // func's default
  kOutput = 1,
  kInout = 2,
  kError = 3, // op's default
};

// util to print
std::string str(ArgSideEffectType argSETy);

// currently use registration-based only
// later, we can iteration-based.
// Note it be override.
struct ArgSideEffectAnalysis {
  ArgSideEffectAnalysis() {}

  virtual ~ArgSideEffectAnalysis() {}

  void addGetType(
      llvm::StringRef name,
      std::function<ArgSideEffectType(mlir::Operation *, unsigned)> check) {
    opNameToGetType.try_emplace(name, check);
  }

  virtual ArgSideEffectType getType(mlir::Operation *op, unsigned argOffset);

  /// Dump the arg side effect information
  void dump(llvm::raw_ostream &os);

  llvm::DenseMap<llvm::StringRef,
                 std::function<ArgSideEffectType(mlir::Operation *, unsigned)>>
      opNameToGetType;
};

} // namespace byteir

#endif // BYTEIR_ANALYSIS_SIDEEFFECT_H