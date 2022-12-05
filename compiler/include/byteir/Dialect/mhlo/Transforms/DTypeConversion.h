//===- DtypeConversion.h -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_DTYPECONVERSION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_DTYPECONVERSION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {
class Operation;
class Value;
class TensorType;
class ModuleOp;

// abstract struct for convert rule
struct DTypeConvertRuleBase {

  DTypeConvertRuleBase(){};
  virtual ~DTypeConvertRuleBase() {}

  // default all function
  virtual bool checkFunc(func::FuncOp) { return true; }

  // default all function
  virtual bool canModifyFuncArg(func::FuncOp) { return false; }

  // Data type rules for operations
  llvm::DenseMap<llvm::StringRef,
                 std::vector<std::pair<std::vector<Type>, std::vector<Type>>>>
      convertRules;
};

// use DTypeConvertRuleBase to decide how to convert data types
std::unique_ptr<OperationPass<ModuleOp>>
createDTypeConversionPass(DTypeConvertRuleBase *);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_DTYPECONVERSION_H
