//===- ConvertInsertion.h -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTINSERTION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTINSERTION_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <memory>
#include <string>

namespace mlir {
class Operation;
class Value;
class TensorType;

// abstract struct for convert rule
struct ConvertRuleBase {

  ConvertRuleBase(){};
  virtual ~ConvertRuleBase() {}

  // default all function
  virtual bool checkFunc(FuncOp) { return true; }

  virtual llvm::Optional<mlir::TensorType> checkType(mlir::Type) = 0;
};

// a common ConvertRule
// using only ElementType in registered convertElementType for checkType
// and only anchorAttr for checkFunc
struct ConvertOnlyCheckElementType : public ConvertRuleBase {

  explicit ConvertOnlyCheckElementType(mlir::StringRef);

  virtual ~ConvertOnlyCheckElementType() {}

  virtual bool checkFunc(FuncOp) override;
  llvm::Optional<mlir::TensorType> checkType(mlir::Type) override;

  // convert element type from first to second
  llvm::DenseMap<mlir::Type, mlir::Type> convertElementType;

  std::string anchorAttr;
};

// use ConvertRuleBase to decide how to insert convert
std::unique_ptr<OperationPass<ModuleOp>>
createConvertInsertionPass(ConvertRuleBase *);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTINSERTION_H