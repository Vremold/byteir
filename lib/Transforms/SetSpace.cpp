//===- SetSpace.cpp -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/SetSpace.h"
#include "./PassDetail.h"
#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <utility>

#define DEBUG_TYPE "set-space-passes"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::memref;

namespace {

using UpdateFuncType_t = std::pair<SmallVector<Type, 4>, SmallVector<Type, 4>>;
using CopyType_t = std::pair<Value, Attribute>;

// utils
const std::string &getSpace(ArrayRef<std::string> spaces, size_t offset) {
  if (offset < spaces.size()) {
    return spaces[offset];
  }
  return spaces.back();
}

bool isEmptyStringAttr(Attribute attr) {
  if (auto strAttr = attr.dyn_cast_or_null<StringAttr>()) {
    return strAttr.strref().empty();
  }
  return false;
}

bool isFuncCorrectSpace(FuncOp func, size_t offset, Attribute space,
                        bool isArg) {
  FunctionType funcType = func.getType();
  Type argType;
  if (isArg) {
    argType = funcType.getInput(offset);
  } else {
    argType = funcType.getResult(offset);
  }

  if (auto memRefTy = argType.dyn_cast<MemRefType>()) {
    return memRefTy.getMemorySpace() == space;
  }
  return false;
}

bool isFuncNotCompatiableWithSpace(FuncOp func, Attribute space) {
  if (auto funcSpaceAttr = func->getAttrOfType<StringAttr>("device")) {
    return funcSpaceAttr != space;
  }

  // func has not space attr
  // check whehther is public
  return func.isPublic();
}

// Maybe change this to bind Module Op later
Attribute getOrCreateSpaceAttr(Operation *op, llvm::StringRef name) {
  return StringAttr::get(op->getContext(), name);
}

// Only implement Non-allow-output-writable
// update function types for args recursively
void updateFuncArgTypes(FuncOp func, ModuleOp m,
                        DenseMap<FuncOp, UpdateFuncType_t> &funcToUpdateTypes,
                        DenseMap<CopyType_t, Value> &copyPairToCopyTargets,
                        size_t offset, Attribute spaceAttr) {
  // skip if suggest spaceAttr is empty
  // or already right space
  if (isEmptyStringAttr(spaceAttr) ||
      isFuncCorrectSpace(func, offset, spaceAttr, true /*isArg*/)) {
    return;
  }

  // initialize funcToUpdateTypes
  if (funcToUpdateTypes.count(func) == 0) {
    FunctionType funcType = func.getType();
    funcToUpdateTypes.try_emplace(
        func,
        SmallVector<Type, 4>(funcType.getInputs().begin(),
                             funcType.getInputs().end()),
        SmallVector<Type, 4>(funcType.getResults().begin(),
                             funcType.getResults().end()));
  }

  auto &newUpdateTypes = funcToUpdateTypes[func];
  // update argType
  auto &argType = newUpdateTypes.first[offset];

  if (auto MemrefTy = argType.dyn_cast<MemRefType>()) {
    auto newArgType = cloneMemRefTypeWithMemSpace(MemrefTy, spaceAttr);
    argType = newArgType;
  }

  if (!func.empty()) {
    Value arg = func.getArgument(offset);
    DenseMap<Attribute, SmallVector<FuncOp, 4>> spaceToCalleeFuncs;
    arg.setType(argType);

    // handle users
    for (auto user : arg.getUsers()) {
      // handle recursive calls
      if (auto callOp = dyn_cast<CallOp>(user)) {
        auto anotherFunc = m.lookupSymbol<FuncOp>(callOp.getCallee());

        // if (isFuncNotCompatiableWithSpace(anotherFunc, spaceAttr)) {
        //}

        if (anotherFunc == nullptr || !anotherFunc.isPrivate()) {
          continue;
        }

        if (auto privateSpaceAttr =
                anotherFunc->getAttrOfType<StringAttr>("device")) {

          if (privateSpaceAttr != spaceAttr) {
            // check this specific exist or not
            CopyType_t copyKey = {arg, privateSpaceAttr};
            FunctionType privateFuncType = anotherFunc.getType();
            for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
              if (arg != callOp.getOperand(i)) {
                continue;
              };

              if (copyPairToCopyTargets.count(copyKey) == 0) {
                // if copy not exist
                // insert alloc and copy
                OpBuilder b(callOp);
                auto loc = callOp.getLoc();
                auto newArg = b.create<memref::AllocOp>(
                    loc, privateFuncType.getInput(i).dyn_cast<MemRefType>());
                b.create<memref::CopyOp>(callOp.getLoc(), arg, newArg);
                copyPairToCopyTargets.try_emplace(copyKey, newArg);
                callOp.setOperand(i, newArg);
              } else {
                // if copy already exist, directly refer it
                auto taget = copyPairToCopyTargets[copyKey];
                callOp.setOperand(i, taget);
              }
            }
          }
        } else {
          // if not specified device, we assume it supports all devices
          // then perform the same space recurively
          for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
            if (arg != callOp.getOperand(i)) {
              continue;
            }
            updateFuncArgTypes(anotherFunc, m, funcToUpdateTypes,
                               copyPairToCopyTargets, i, spaceAttr);
          }
        }
      }
    }
  }
}

// Only implement Non-allow-output-writable
// update function types for return types recursively
void updateFuncReturnTypes(
    FuncOp func, ModuleOp m,
    DenseMap<FuncOp, UpdateFuncType_t> &funcToUpdateTypes,
    DenseMap<CopyType_t, Value> &copyPairToCopyTargets, size_t offset,
    Attribute spaceAttr) {
  // skip if suggest spaceAttr is empty
  // or already right space
  if (isEmptyStringAttr(spaceAttr) ||
      isFuncCorrectSpace(func, offset, spaceAttr, false /*isArg*/)) {
    return;
  }

  // initialize funcToUpdateTypes
  if (funcToUpdateTypes.count(func) == 0) {
    FunctionType funcType = func.getType();
    funcToUpdateTypes.try_emplace(
        func,
        SmallVector<Type, 4>(funcType.getInputs().begin(),
                             funcType.getInputs().end()),
        SmallVector<Type, 4>(funcType.getResults().begin(),
                             funcType.getResults().end()));
  }

  auto &newUpdateTypes = funcToUpdateTypes[func];
  // update retType
  auto &retType = newUpdateTypes.second[offset];

  if (auto MemrefTy = retType.dyn_cast<MemRefType>()) {
    auto newRetType = cloneMemRefTypeWithMemSpace(MemrefTy, spaceAttr);
    retType = newRetType;
  }

  if (!func.empty()) {
    func::ReturnOp retOp = *func.getOps<func::ReturnOp>().begin();
    Value ret = retOp.getOperand(offset);

    if (auto callOp = ret.getDefiningOp<CallOp>()) {
      // handle return as a call's results
      if (auto anotherFunc = m.lookupSymbol<FuncOp>(callOp.getCallee())) {

        if (isFuncNotCompatiableWithSpace(anotherFunc, spaceAttr)) {
          // insert copy after
          OpBuilder b(callOp);
          b.setInsertionPointAfter(callOp);
          auto loc = callOp.getLoc();
          FunctionType privateFuncType = anotherFunc.getType();

          for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
            if (ret != callOp.getResult(i)) {
              continue;
            }

            auto newRet =
                b.create<memref::AllocOp>(loc, retType.dyn_cast<MemRefType>());
            ret.replaceAllUsesWith(newRet);
            b.create<memref::CopyOp>(callOp.getLoc(), ret, newRet);
            ret = newRet;
            break;
          }

        } else {
          // compatiable function
          ret.setType(retType);
          for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
            if (ret != callOp.getResult(i)) {
              continue;
            }
            updateFuncReturnTypes(anotherFunc, m, funcToUpdateTypes,
                                  copyPairToCopyTargets, i, spaceAttr);
            break;
          }
        }
      }
    } else {
      // regular alloc
      LLVM_DEBUG(llvm::dbgs()
                 << "arg is modified in " << func.getName() << "\n");
    }
  }
}

struct SetAllSpacePass : public SetAllSpaceBase<SetAllSpacePass> {
  explicit SetAllSpacePass() = default;

  SetAllSpacePass(std::string entryFuncName, const std::string &space_)
      : SetAllSpaceBase() {
    entryFunc = entryFuncName;
    space = space_;
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

    auto ctx = m->getContext();
    auto newSpace = StringAttr::get(ctx, space);

    DenseMap<FuncOp, UpdateFuncType_t> funcToArgTypes;
    DenseMap<CopyType_t, Value> copyPairToCopyTargets;

    // resolve entry function
    // argumenets
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      updateFuncArgTypes(funcOp, m, funcToArgTypes, copyPairToCopyTargets, i,
                         newSpace);
    }

    // results
    for (unsigned i = 0, e = funcOp.getNumResults(); i < e; ++i) {
      updateFuncReturnTypes(funcOp, m, funcToArgTypes, copyPairToCopyTargets, i,
                            newSpace);
    }

    // local alloc
    for (auto alloc : funcOp.getOps<memref::AllocOp>()) {
      auto ret = alloc.getResult();
      if (auto MemrefTy = ret.getType().dyn_cast<MemRefType>()) {
        auto newRetType = cloneMemRefTypeWithMemSpace(MemrefTy, newSpace);
        ret.setType(newRetType);
      }
    }

    // rewrite FunctionType
    for (auto it : funcToArgTypes) {
      FunctionType funcType = it.first.getType();
      it.first.setType(
          FunctionType::get(ctx, it.second.first, it.second.second));
    }
  }
};

struct SetArgSpacePass : public SetArgSpaceBase<SetArgSpacePass> {

  explicit SetArgSpacePass() = default;

  SetArgSpacePass(std::string entryFuncName, const std::string &space,
                  bool allowOutWritable)
      : SetArgSpaceBase() {
    entryFunc = entryFuncName;
    allSpace = space;
    allowArgWritable = allowOutWritable;
  }

  SetArgSpacePass(std::string entryFuncName, ArrayRef<std::string> argList,
                  ArrayRef<std::string> retList, bool allowOutWritable)
      : SetArgSpaceBase(), argSpaces(argList.begin(), argList.end()),
        retSpaces(retList.begin(), retList.end()) {
    entryFunc = entryFuncName;
    allowArgWritable = allowOutWritable;
  }

  void runOnOperation() override {
    // TODO: after supporting allowArgWritable version, remove this.
    if (allowArgWritable) {
      LLVM_DEBUG(llvm::dbgs()
                 << "allowArgWritable version is not implmented yet.\n");
    }

    // early termination
    if (entryFunc.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No speficied function.\n");
      return;
    }

    ModuleOp m = getOperation();
    auto ctx = m.getContext();
    FuncOp funcOp = m.lookupSymbol<FuncOp>(entryFunc);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot find the speficied function "
                              << entryFunc.getValue() << "\n");
      return;
    }

    // parse spaces
    if (argSpaces.empty() && !allSpace.getValue().empty()) {
      argSpaces.push_back(allSpace.getValue());
    }

    if (argSpaces.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No speficied argSpaces.\n");
      return;
    }

    if (retSpaces.empty() && !allSpace.getValue().empty()) {
      retSpaces.push_back(allSpace.getValue());
    }

    if (retSpaces.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No speficied retSpaces.\n");
      return;
    }

    DenseMap<FuncOp, UpdateFuncType_t> funcToArgTypes;
    DenseMap<CopyType_t, Value> copyPairToCopyTargets;

    // resolve entry function
    // argumenets
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      auto newSpace = getOrCreateSpaceAttr(m, getSpace(argSpaces, i));
      updateFuncArgTypes(funcOp, m, funcToArgTypes, copyPairToCopyTargets, i,
                         newSpace);
    }
    // results
    for (unsigned i = 0, e = funcOp.getNumResults(); i < e; ++i) {
      auto newSpace = getOrCreateSpaceAttr(m, getSpace(retSpaces, i));
      updateFuncReturnTypes(funcOp, m, funcToArgTypes, copyPairToCopyTargets, i,
                            newSpace);
    }

    // resolve device functions
    for (auto deviceFuncOp : m.getOps<FuncOp>()) {

      // skip non-private or one without device attr
      if (!deviceFuncOp.isPrivate() ||
          !deviceFuncOp->hasAttrOfType<StringAttr>("device")) {
        continue;
      }

      auto deviceAttr = deviceFuncOp->getAttrOfType<StringAttr>("device");

      // argumenets
      for (unsigned i = 0, e = deviceFuncOp.getNumArguments(); i < e; ++i) {
        updateFuncArgTypes(deviceFuncOp, m, funcToArgTypes,
                           copyPairToCopyTargets, i, deviceAttr);
      }

      // results
      for (unsigned i = 0, e = deviceFuncOp.getNumResults(); i < e; ++i) {
        updateFuncReturnTypes(deviceFuncOp, m, funcToArgTypes,
                              copyPairToCopyTargets, i, deviceAttr);
      }

      // handle callee
      auto &newArgTypes = funcToArgTypes[deviceFuncOp];
      auto maybeSymbolUses = deviceFuncOp.getSymbolUses(m);
      for (SymbolTable::SymbolUse symbolUse : *maybeSymbolUses) {
        if (auto callOp = dyn_cast<CallOp>(symbolUse.getUser())) {
          // arguement
          for (unsigned i = 0, e = callOp.getNumOperands(); i < e; ++i) {
            callOp.getOperand(i).setType(newArgTypes.first[i]);
          }

          // resultss
          for (unsigned i = 0, e = callOp.getNumResults(); i < e; ++i) {
            callOp.getResult(i).setType(newArgTypes.second[i]);
          }
        }
      }
    }

    // rewrite FunctionType

    for (auto it : funcToArgTypes) {
      FunctionType funcType = it.first.getType();
      it.first.setType(
          FunctionType::get(ctx, it.second.first, it.second.second));
    }
  }

  llvm::SmallVector<std::string, 16> argSpaces;
  llvm::SmallVector<std::string, 16> retSpaces;
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createSetAllSpacePass(std::string entryFunc, const std::string &space) {
  return std::make_unique<SetAllSpacePass>(entryFunc, space);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createSetArgSpacePass(std::string entryFunc, const std::string &allSpace,
                            bool allowArgWritable) {
  return std::make_unique<SetArgSpacePass>(entryFunc, allSpace,
                                           allowArgWritable);
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createSetArgSpacePass(
    std::string entryFunc, llvm::ArrayRef<std::string> argSpaces,
    llvm::ArrayRef<std::string> retSpaces, bool allowArgWritable) {
  return std::make_unique<SetArgSpacePass>(entryFunc, argSpaces, retSpaces,
                                           allowArgWritable);
}
