//===- FunctionSupport.cpp ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/Common/FunctionSupport.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace llvm;

namespace mlir {
namespace function_interface_impl {
// llvm upstream insertFunctionArgument cannot handle inserting multiple args
static inline void
insertFunctionArgumentsEx(Operation *op, ArrayRef<unsigned> argIndices,
                          TypeRange argTypes, ArrayRef<DictionaryAttr> argAttrs,
                          ArrayRef<Location> argLocs, unsigned originalNumArgs,
                          Type newType) {
  assert(argIndices.size() == argTypes.size());
  assert(argIndices.size() == argAttrs.size() || argAttrs.empty());
  assert(argIndices.size() == argLocs.size() || argLocs.empty());
  if (argIndices.empty())
    return;

  // There are 3 things that need to be updated:
  // - Function type.
  // - Arg attrs.
  // - Block arguments of entry block.
  Block &entry = op->getRegion(0).front();

  // Update the argument attributes of the function.
  auto oldArgAttrs = op->getAttrOfType<ArrayAttr>(getArgDictAttrName());
  if (oldArgAttrs || !argAttrs.empty()) {
    SmallVector<DictionaryAttr, 4> newArgAttrs;
    newArgAttrs.reserve(originalNumArgs + argIndices.size());
    unsigned oldIdx = 0;
    auto migrate = [&](unsigned oldNum) {
      if (!oldArgAttrs) {
        newArgAttrs.resize(newArgAttrs.size() + oldNum);
      } else {
        auto oldArgAttrRange = oldArgAttrs.getAsRange<DictionaryAttr>();
        auto newIdx = oldIdx + oldNum;
        newArgAttrs.append(oldArgAttrRange.begin() + oldIdx,
                           oldArgAttrRange.begin() + newIdx);
        oldIdx = newIdx;
      }
    };
    for (unsigned i = 0, e = argIndices.size(); i < e; ++i) {
      migrate(i ? argIndices[i] - argIndices[i - 1] - 1 : argIndices[i]);
      newArgAttrs.push_back(argAttrs.empty() ? DictionaryAttr{} : argAttrs[i]);
    }
    migrate(originalNumArgs - oldIdx);
    setAllArgAttrDicts(op, newArgAttrs);
  }

  // Update the function type and any entry block arguments.
  op->setAttr(getTypeAttrName(), TypeAttr::get(newType));
  for (unsigned i = 0, e = argIndices.size(); i < e; ++i)
    entry.insertArgument(argIndices[i], argTypes[i],
                         argLocs.empty() ? op->getLoc() : argLocs[i]);
}
} // namespace function_interface_impl
} // namespace mlir

//
// LWC NOTE This implementation DO NOT support inout,
// meaning directly returning an input as an results
// LWC NOTE Also DO NOT support duplicated results.
//
namespace {

static inline void replicateFuncOpResultSigature(func::FuncOp funcOp) {
  mlir::FunctionType oldFuncType = funcOp.getFunctionType();

  llvm::SmallVector<Type, 16> newInputTypes(oldFuncType.getInputs().begin(),
                                            oldFuncType.getInputs().end());
  newInputTypes.append(oldFuncType.getResults().begin(),
                       oldFuncType.getResults().end());

  unsigned origianlSize = oldFuncType.getInputs().size();
  llvm::SmallVector<unsigned, 16> relocatedIndices;
  relocatedIndices.reserve(oldFuncType.getNumResults());
  for (unsigned int i = 0; i < oldFuncType.getNumResults(); ++i) {
    relocatedIndices.push_back(origianlSize + i);
  }

  mlir::OpBuilder opBuilder(funcOp);

  mlir::FunctionType newFuncType =
      opBuilder.getFunctionType(newInputTypes, {} /*results*/);

  llvm::SmallVector<DictionaryAttr, 4> resultAttrs;
  for (size_t i = 0; i < funcOp.getNumResults(); ++i) {
    NamedAttrList attrList = funcOp.getResultAttrs(i);
    resultAttrs.push_back(attrList.getDictionary(funcOp->getContext()));
  }
  funcOp->removeAttr(function_interface_impl::getResultDictAttrName());

  mlir::function_interface_impl::insertFunctionArgumentsEx(
      funcOp, relocatedIndices, oldFuncType.getResults(), resultAttrs, {},
      origianlSize, newFuncType);
  //  Note the FuncOp's member function seems buggy. Use the above
  //  instread. funcOp.insertArguments(relocatedIndices,
  //  oldFuncType.getResults(), {},
  //  {});
}
} // namespace

void mlir::replicateFuncOpResults(func::FuncOp funcOp) {
  unsigned idx = funcOp.getNumArguments();

  replicateFuncOpResultSigature(funcOp);

  if (funcOp.empty()) {
    return;
  }

  func::ReturnOp retOp =
      cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());

  llvm::SmallPtrSet<mlir::Operation *, 16> retAllocs;

  for (auto retVal : retOp.getOperands()) {
    retVal.replaceAllUsesExcept(funcOp.getArgument(idx++), retOp);
    if (auto allocOp = dyn_cast<memref::AllocOp>(retVal.getDefiningOp())) {
      if (retAllocs.count(allocOp) == 0) {
        retAllocs.insert(allocOp);
      }
    }
  }

  mlir::OpBuilder opBuilder(retOp);
  opBuilder.create<func::ReturnOp>(retOp.getLoc());
  retOp.erase();
  for (auto op : retAllocs) {
    op->erase();
  }
}

void mlir::replicateFuncOpResults(
    func::FuncOp funcOp, std::function<void(func::ReturnOp)> retOpHandling) {
  replicateFuncOpResultSigature(funcOp);

  if (funcOp.empty()) {
    return;
  }

  func::ReturnOp retOp =
      cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());
  retOpHandling(retOp);
}

void mlir::relocateFuncOpConstantLike(
    func::FuncOp funcOp, std::function<bool(mlir::Operation *)> checkOp,
    std::function<std::tuple<mlir::Value, NamedAttrList>(mlir::Operation *)>
        getValue) {
  // skip empty func
  if (funcOp.empty())
    return;

  mlir::FunctionType oldFuncType = funcOp.getFunctionType();

  llvm::SmallVector<Type, 16> relocatedTypes;

  unsigned origianlSize = oldFuncType.getInputs().size();
  llvm::SmallVector<unsigned, 16> relocatedIndices;

  // might use DenseSet instread if constant is too many
  // llvm::SmallPtrSet<mlir::Operation*, 16> consantLikes;
  llvm::SmallPtrSet<mlir::Value, 16> consantLikeValues;
  llvm::SmallVector<mlir::Operation *, 16> constantLikeOps;
  llvm::SmallVector<DictionaryAttr, 16> correspondingArgAttrs;

  // collect all constantLikes
  unsigned offset = 0;
  funcOp.walk([&](mlir::Operation *op) {
    if (checkOp(op)) {
      auto t = getValue(op);
      mlir::Value &val = std::get<0>(t);
      NamedAttrList &attrList = std::get<1>(t);
      consantLikeValues.insert(val);
      constantLikeOps.push_back(op);
      relocatedIndices.push_back(offset);
      relocatedTypes.push_back(val.getType());
      correspondingArgAttrs.push_back(attrList.getDictionary(op->getContext()));
      offset++;
    }
  });

  // if no weight
  if (offset == 0) {
    return;
  }

  llvm::SmallVector<Type, 16> newInputTypes = relocatedTypes; // copy
  newInputTypes.append(oldFuncType.getInputs().begin(),
                       oldFuncType.getInputs().end());

  mlir::OpBuilder opBuilder(funcOp);

  mlir::FunctionType newFuncType =
      opBuilder.getFunctionType(newInputTypes, {} /*results*/);

  mlir::function_interface_impl::insertFunctionArgumentsEx(
      funcOp, relocatedIndices, relocatedTypes, correspondingArgAttrs, {},
      origianlSize, newFuncType);

  unsigned idx = 0;
  for (auto val : consantLikeValues) {
    val.replaceAllUsesWith(funcOp.getArgument(relocatedIndices[idx]));
    Operation *op = constantLikeOps[idx++];
    if (val.getDefiningOp() != nullptr) {
      if (op != val.getDefiningOp()) {
        op->erase();
      }
      val.getDefiningOp()->erase();
    }
  }
}