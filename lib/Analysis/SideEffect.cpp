//===- SideEffect.cpp -----------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Analysis/SideEffect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace byteir;
using namespace llvm;
using namespace mlir;
using namespace mlir::func;

std::string byteir::to_str(ArgSideEffectType argSETy) {
  if (argSETy == ArgSideEffectType::kInput) {
    return "kInput";
  }

  if (argSETy == ArgSideEffectType::kOutput) {
    return "kOutput";
  }

  if (argSETy == ArgSideEffectType::kInout) {
    return "kInout";
  }

  if (argSETy == ArgSideEffectType::kError) {
    return "kError";
  }

  return "";
}

void ArgSideEffectAnalysis::dump(raw_ostream &os) {
  os << "============= registry of arg side effect"
     << " =============\n";
  for (auto it : opNameToGetType) {
    os << it.first << "\n";
  }
}

ArgSideEffectType ArgSideEffectAnalysis::getType(mlir::Operation *op,
                                                 unsigned argOffset) {
  if (op == nullptr) {
    return ArgSideEffectType::kError;
  }

  // if having MemoryEffectOpInterface, use it first
  if (auto opInter = dyn_cast<MemoryEffectOpInterface>(op)) {
    if (opInter.getEffectOnValue<MemoryEffects::Read>(op->getOperand(argOffset))
            .hasValue()) {
      return ArgSideEffectType::kInput;
    }
    return ArgSideEffectType::kOutput;
  }

  // if no MemoryEffectOpInterface, use lookup
  // check call first, use fucOp's name in lookup
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    StringRef name = callOp.getCallee();

    if (opNameToGetType.count(name) == 0) {
      // return input by default for a call
      return ArgSideEffectType::kInput;
    }

    return opNameToGetType[name](op, argOffset);
  }

  StringRef name = op->getName().getStringRef();

  if (opNameToGetType.count(name) == 0) {
    // return error by default for an op
    return ArgSideEffectType::kError;
  }

  return opNameToGetType[name](op, argOffset);
}