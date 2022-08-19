//===- ConvertFuncToCustomCall.h ------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTFUNCTOCUSTOMCALL_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTFUNCTOCUSTOMCALL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <string>

namespace mlir {
class ModuleOp;
class NamedAttrList;
class Value;
class ValueRange;
class TypeRange;

// the abstract class of FuncToCustomCallConverter
// Some member functions are implemented for trival cases
struct FuncToCustomCallConverterBase {
  FuncToCustomCallConverterBase(){};
  virtual ~FuncToCustomCallConverterBase() {}

  virtual bool checkFunc(func::FuncOp) = 0;

  virtual NamedAttrList getAttrs(func::FuncOp) = 0;

  virtual TypeRange getResultTypes(func::FuncOp func) {
    return func.getResultTypes();
  }

  virtual ValueRange getOperands(func::CallOp call) {
    return call.getOperands();
  }

  virtual unsigned getNewResultIdx(func::CallOp, unsigned oldIdx) {
    return oldIdx;
  }
};

// a common CustomMeta for creating CustomCall
struct CustomLoopupMeta {
  std::string callTargetName;
  bool hasSideEffect;
  bool useDefault;
  SmallVector<unsigned> opernadOldIndices; // new id to old id
  SmallVector<unsigned> resultOldIndices;  // new id to old id
  SmallVector<unsigned> resultNewIndices;  // old id to new id

  CustomLoopupMeta()
      : callTargetName(""), hasSideEffect(false), useDefault(true) {}

  CustomLoopupMeta(const std::string &tagetName, bool sideEffect)
      : callTargetName(tagetName), hasSideEffect(sideEffect), useDefault(true) {
  }

  CustomLoopupMeta(const std::string &tagetName, bool sideEffect,
                   ArrayRef<unsigned> operand, ArrayRef<unsigned> resultOld,
                   ArrayRef<unsigned> resultNew)
      : callTargetName(tagetName), hasSideEffect(sideEffect), useDefault(false),
        opernadOldIndices(operand.begin(), operand.end()),
        resultOldIndices(resultOld.begin(), resultOld.end()),
        resultNewIndices(resultNew.begin(), resultNew.end()) {}
};

// a common FuncToCustomCallConverter using Lookup
struct FuncToCustomCallConverterLookup : public FuncToCustomCallConverterBase {

  FuncToCustomCallConverterLookup() : FuncToCustomCallConverterBase() {}

  explicit FuncToCustomCallConverterLookup(
      const llvm::StringMap<CustomLoopupMeta> &externalMap)
      : FuncToCustomCallConverterBase(), funcNameToCustomMeta(externalMap) {}

  virtual ~FuncToCustomCallConverterLookup() {}

  bool checkFunc(func::FuncOp) override;

  NamedAttrList getAttrs(func::FuncOp) override;

  TypeRange getResultTypes(func::FuncOp) override;

  ValueRange getOperands(func::CallOp) override;

  unsigned getNewResultIdx(func::CallOp, unsigned) override;

  llvm::StringMap<CustomLoopupMeta> funcNameToCustomMeta;
};

std::unique_ptr<OperationPass<ModuleOp>>
createConvertFuncToCustomCallPass(FuncToCustomCallConverterBase *converter);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTFUNCTOCUSTOMCALL_H