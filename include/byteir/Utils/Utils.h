//===- Utils.h ------------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_H
#define BYTEIR_UTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/FunctionSupport.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <type_traits>

namespace mlir {
class CallOp;
class FuncOp;
class Operation;
class Value;

// Return true when a value is a ConstantIndex with value of `lit`.
bool isConstantIndex(Value value, int64_t lit);

// Check whether an attribute is zero
// If an attribute contain multiple sub attributes,
// it will check all of sub attributes.
bool isZeroAttribute(Attribute value);

// Returns true if the given `attr` is a splat value and is `value`.
bool isSplatValue(DenseIntElementsAttr attr, int64_t value);

// Returns true if the given `attr` is a splat value and close to `value`.
bool isSplatCloseToValue(DenseFPElementsAttr attr, double value,
                         double EPSILON = 0.00001);

// extract values in `attr` to `arrayValues`
void getValuesFromDenseIntElementsAttr(DenseIntElementsAttr attr,
                                       SmallVector<int64_t> &arrayValues);

// Return a placeholder name of an attribute
// to avoid breaking the verifier of the original attribute
// by adding some unique prefix or postfix
std::string getAttrPlaceholderName(StringRef name);

// Remove placeholder of attribute names
// Note: it removes placeholder tag only.
// The attribute name will become the original name
// Note: the input is a list of original names
void removeAttrPlaceholders(mlir::Operation *op, ArrayRef<StringRef> Orignames);

// Remove placeholder of arg attribute names
// Note: it removes placeholder tag only.
// The attribute name will become the original name
// Note: the input is a list of original arg attribute names
template <typename OpTy>
std::enable_if_t<OpTy::template hasTrait<OpTrait::FunctionLike>()>
removeArgAttrPlaceholders(OpTy op, ArrayRef<StringRef> argAttrNames) {
  for (size_t idx = 0; idx < op.getNumArguments(); ++idx) {
    for (const auto &name : argAttrNames) {
      auto placeholder = getAttrPlaceholderName(name);
      auto attr = op.getArgAttr(idx, placeholder);
      if (attr == nullptr) {
        continue;
      }

      op.setArgAttr(idx, name, attr);
      op.removeArgAttr(idx, placeholder);
    }
  }
}

// Return FuncOp from a CallOp
mlir::FuncOp GetFuncOp(mlir::CallOp);

// Return true if attrs has any of filterAttrs
bool HasAnyOfAttrs(llvm::ArrayRef<mlir::NamedAttribute> attrs,
                   llvm::ArrayRef<llvm::StringRef> filterAttrs);

void AddAttrs(mlir::Operation *, llvm::ArrayRef<mlir::NamedAttribute> attrs);

Optional<unsigned> FindOperandIndex(mlir::Operation *, mlir::Value);

Optional<unsigned> FindResultIndex(mlir::Operation *, mlir::Value);

SmallVector<Value, 4>
GetInputsOfCluster(const llvm::SmallVector<Operation *, 8> &cluster);

SmallVector<Value, 4>
GetOutputsOfCluster(const llvm::SmallVector<Operation *, 8> &cluster);

// return true, if memref is only used in op in the filters, or alloc or dealloc
bool IsMemrefTrivial(mlir::Value memref, llvm::ArrayRef<mlir::Operation*> filters);

} // namespace mlir

#endif // BYTEIR_UTILS_H