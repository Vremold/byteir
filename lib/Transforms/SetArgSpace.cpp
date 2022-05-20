//===- SetArgSpace.cpp ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/SetArgSpace.h"
#include "./PassDetail.h"
#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "set-arg-space-pass"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::memref;

namespace {

// utils
const std::string &getSpace(ArrayRef<std::string> spaces, size_t offset) {
  if (offset < spaces.size()) {
    return spaces[offset];
  }
  return spaces.back();
}

// TODO change this to bind Module Op
Attribute getOrcreateSpaceAttr(Operation *op, llvm::StringRef name) {
  return StringAttr::get(op->getContext(), name);
}

// update function types recursively
void updateFuncArgTypes(FuncOp func, ModuleOp m,
                        DenseMap<FuncOp, SmallVector<Type, 4>> &funcToArgs,
                        size_t offset, Attribute spaceAttr) {
  if (funcToArgs.count(func) == 0) {
    FunctionType funcType = func.getType();
    funcToArgs.try_emplace(func,
                           SmallVector<Type, 4>(funcType.getInputs().begin(),
                                                funcType.getInputs().end()));
  }

  auto &newArgTypes = funcToArgs[func];

  // update argType
  auto &argType = newArgTypes[offset];
  if (auto MemrefTy = argType.dyn_cast<MemRefType>()) {
    auto newArgType = cloneMemRefTypeWithMemSpace(MemrefTy, spaceAttr);
    argType = newArgType;
  }

  if (!func.empty()) {
    auto arg = func.getArgument(offset);
    arg.setType(newArgTypes[offset]);

    DenseMap<Attribute, SmallVector<FuncOp, 4>> spaceToCalleeFuncs;

    for (auto user : arg.getUsers()) {
      if (auto callOp = dyn_cast<CallOp>(user)) {
        auto privateFunc = m.lookupSymbol<FuncOp>(callOp.getCallee());
        if (privateFunc == nullptr || !privateFunc.isPrivate()) {
          continue;
        }

        if (auto privateSpaceAttr =
                privateFunc->getAttrOfType<StringAttr>("device")) {

          if (privateSpaceAttr != spaceAttr) {
            // rewrite it by inserting alloc and copy
            OpBuilder b(callOp);
            auto loc = callOp.getLoc();
            FunctionType privateFuncType = privateFunc.getType();
            for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
              if (arg != callOp.getOperand(i)) {
                continue;
              };

              auto newArg = b.create<memref::AllocOp>(
                  loc, privateFuncType.getInput(i).dyn_cast<MemRefType>());
              b.create<memref::CopyOp>(callOp.getLoc(), arg, newArg);
              callOp.setOperand(i, newArg);
            }
          }

        } else {
          // if not specified device, we assume it supports all devices
          // then perform the same space recurively
          for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
            if (arg != callOp.getOperand(i)) {
              continue;
            }
            updateFuncArgTypes(privateFunc, m, funcToArgs, i, spaceAttr);
          }
        }
      }
    }
  }
}

struct SetArgSpacePass : public SetArgSpaceBase<SetArgSpacePass> {

  explicit SetArgSpacePass() = default;

  SetArgSpacePass(std::string entryFuncName, std::string space)
      : SetArgSpaceBase() {
    entryFunc = entryFuncName;
    allSpace = space;
  }

  SetArgSpacePass(std::string entryFuncName, ArrayRef<std::string> spaceList)
      : SetArgSpaceBase(), spaces(spaceList.begin(), spaceList.end()) {
    entryFunc = entryFuncName;
  }

  void runOnOperation() override {
    // early termination
    if (entryFunc.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No speficied function.\n");
      return;
    }

    ModuleOp m = getOperation();
    FuncOp funcOp = m.lookupSymbol<FuncOp>(entryFunc);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot find the speficied function "
                              << entryFunc.getValue() << "\n");
      return;
    }

    // parse spaces
    if (spaces.empty() && !allSpace.getValue().empty()) {
      spaces.push_back(allSpace.getValue());
    }

    if (spaces.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No speficied space.\n");
      return;
    }

    DenseMap<FuncOp, SmallVector<Type, 4>> funcToArgTypes;

    // resolve entry function
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      auto newSpace = getOrcreateSpaceAttr(m, getSpace(spaces, i));
      updateFuncArgTypes(funcOp, m, funcToArgTypes, i, newSpace);
    }

    // resolve device functions
    for (auto deviceFuncOp : m.getOps<FuncOp>()) {

      // skip non-private or one without device attr
      if (!deviceFuncOp.isPrivate() ||
          !deviceFuncOp->hasAttrOfType<StringAttr>("device")) {
        continue;
      }

      auto deviceAttr = deviceFuncOp->getAttrOfType<StringAttr>("device");

      for (unsigned i = 0, e = deviceFuncOp.getNumArguments(); i < e; ++i) {
        updateFuncArgTypes(deviceFuncOp, m, funcToArgTypes, i, deviceAttr);
      }

      // handle callee
      auto &newArgTypes = funcToArgTypes[deviceFuncOp];
      auto maybeSymbolUses = deviceFuncOp.getSymbolUses(m);
      for (SymbolTable::SymbolUse symbolUse : *maybeSymbolUses) {
        if (auto callOp = dyn_cast<CallOp>(symbolUse.getUser())) {
          for (unsigned i = 0, e = callOp.getNumOperands(); i < e; ++i) {
            callOp.getOperand(i).setType(newArgTypes[i]);
          }
        }
      }
    }

    // rewrite FunctionType
    auto ctx = m.getContext();
    for (auto it : funcToArgTypes) {
      FunctionType funcType = it.first.getType();
      it.first.setType(
          FunctionType::get(ctx, it.second, funcType.getResults()));
    }
  }

  llvm::SmallVector<std::string> spaces;
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createSetArgSpacePass(std::string entryFunc, std::string allSpace) {
  return std::make_unique<SetArgSpacePass>(entryFunc, allSpace);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createSetArgSpacePass(std::string entryFunc,
                            llvm::ArrayRef<std::string> spaces) {
  return std::make_unique<SetArgSpacePass>(entryFunc, spaces);
}
