//===- Utils.h ------------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_UTILS_H
#define BYTEIR_UTILS_UTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <type_traits>

namespace mlir {
class Location;
class Operation;
class Value;

namespace func {
class CallOp;
class FuncOp;
} // namespace func

// Return literal from a constant-like value
// return None if not applicable
llvm::Optional<int64_t> getLiteralFromConstantLike(Value);

// Return literal from a constant-like value,
// return defaultLit if not applicable
int64_t getLiteralFromConstantLike(Value, int64_t defaultLit);

// Return literals from a list of constant-like values
llvm::SmallVector<int64_t, 4>
getLiteralsFromConstantLikes(ArrayRef<Value> values, int64_t defaultLit);

// Create a vector with only the offset as 1, the rest as 0's.
// e.g. if offset == 1, size == 4, val == 3, return3 [0, 3, 0, 0]
llvm::SmallVector<int64_t, 4> createOneHot(unsigned size, unsigned offset,
                                           int64_t val = 1);

// Return all indices for non-zeros
llvm::SmallVector<unsigned, 4> getAllIndicesForNonZeros(ArrayRef<int64_t>);

// Return true when a value is a ConstantIndex with value of `lit`.
bool isConstantIndex(Value value, int64_t lit);

// Check whether an attribute is zero
// If an attribute contain multiple sub attributes,
// it will check all of sub attributes.
bool isZeroAttribute(Attribute value);

// Check whether an attribute is SmallestAttr
// If an attribute contain multiple sub attributes,
// it will check all of sub attributes.
bool isMinValueAttribute(Attribute value);

// TODO add Largest if needed.

// Returns true if the given `attr` is a splat value and is `value`.
bool isSplatValue(DenseIntElementsAttr attr, int64_t value);

// Returns true if the given `attr` is a splat value as the given `value`.
bool isSplatValue(DenseFPElementsAttr attr, double value);

// Returns true if the given `attr` is a splat value and close to `value`.
bool isSplatCloseToValue(DenseFPElementsAttr attr, double value,
                         double EPSILON = 0.00001);

// extract values in `attr` to `arrayValues`
void getValuesFromDenseIntElementsAttr(DenseIntElementsAttr attr,
                                       SmallVector<int64_t> &arrayValues);

// Returns nD 64-bit dense elements attribute with the given values.
inline DenseIntElementsAttr getI64ElementsAttr(ArrayRef<int64_t> values,
                                               ArrayRef<int64_t> shape,
                                               Builder *builder) {
  RankedTensorType ty =
      RankedTensorType::get(shape, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, values);
}

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
std::enable_if_t<OpTy::template hasTrait<FunctionOpInterface::Trait>()>
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
mlir::func::FuncOp getFuncOp(func::CallOp);

// Return true if attrs has any of filterAttrs
bool hasAnyOfAttrs(llvm::ArrayRef<mlir::NamedAttribute> attrs,
                   llvm::ArrayRef<llvm::StringRef> filterAttrs);

// add `attrs` into an operation
void addAttrs(mlir::Operation *, llvm::ArrayRef<mlir::NamedAttribute> attrs);

Optional<unsigned> findOperandIndex(mlir::Operation *, mlir::Value);

Optional<unsigned> findResultIndex(mlir::Operation *, mlir::Value);

SmallVector<Value, 4>
getInputsOfCluster(const llvm::SmallVector<Operation *, 8> &cluster);

SmallVector<Value, 4> getOutputsOfCluster(
    const llvm::SmallVector<Operation *, 8> &cluster,
    const llvm::DenseMap<Value, int64_t> *outputStats = nullptr);

// return true, if memref is only used in op in the filters, or alloc or dealloc
bool isMemrefTrivial(mlir::Value memref,
                     llvm::ArrayRef<mlir::Operation *> filters);

// count number of a value is used
// if a value is used twice by a user, it will count twice
inline int useCount(Value val) {
  return static_cast<int>(
      std::distance(val.getUses().begin(), val.getUses().end()));
}
// count number of users a value has
// if a value is used twice by a user, it will count one
int userCount(Value val);

inline Location getFusedLoc(ArrayRef<Operation *> ops, OpBuilder &op_builder) {
  llvm::SmallVector<Location> locs;
  for (auto op : ops) {
    locs.push_back(op->getLoc());
  }
  return op_builder.getFusedLoc(locs);
}

} // namespace mlir

#endif // BYTEIR_UTILS_UTILS_H