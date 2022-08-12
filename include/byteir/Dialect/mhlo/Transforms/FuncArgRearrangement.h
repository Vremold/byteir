//===- FuncArgRearrangement.h ---------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_FUNCARGREARRANGEMENT_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_FUNCARGREARRANGEMENT_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
class FunctionType;
class ModuleOp;
class OpBuilder;
class Value;

namespace func {
class FuncOp;
} // namespace func

// abstract base class of FuncArgRearranger
class FuncArgRearrangerBase {
public:
  FuncArgRearrangerBase() {}

  virtual ~FuncArgRearrangerBase() {}

  virtual bool init() = 0;

  virtual FunctionType getFunctionType() = 0;

  // get or create a FuncArg
  virtual Value getOrCreateNewFromOldFuncArg(OpBuilder &b, unsigned newId,
                                             ArrayRef<Value> oldValues) = 0;

  // note old arg might not be constructed from new ones
  virtual llvm::Optional<Value>
  getOrCreateOldFromNewFuncArg(OpBuilder &b, unsigned oldId,
                               ArrayRef<Value> newValues) = 0;

  // get or create a FuncResult
  virtual Value getOrCreateNewFromOldFuncResult(OpBuilder &b, unsigned newId,
                                                ArrayRef<Value> oldValues) = 0;

  // note old result might not be constructed from new ones
  virtual llvm::Optional<Value>
  getOrCreateOldFromNewFuncResult(OpBuilder &b, unsigned oldId,
                                  ArrayRef<Value> newValues) = 0;
};

// abstract base class of FuncArgRearrangerBuilder
class FuncArgRearrangerBuilderBase {
public:
  FuncArgRearrangerBuilderBase() {}
  virtual ~FuncArgRearrangerBuilderBase() {}

  virtual std::unique_ptr<FuncArgRearrangerBase>
  createFuncArgRearranger(func::FuncOp f) = 0;
};

std::unique_ptr<OperationPass<ModuleOp>>
createFuncArgRearrangementPass(FuncArgRearrangerBuilderBase *builder,
                               const std::string &anchor = "");

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_FUNCARGREARRANGEMENT_H
