//===- Utils.cpp ----------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include <assert.h>

using namespace llvm;
using namespace mlir;

int64_t mlir::getLiteralFromConstantLike(Value v, int64_t defaultLit) {
  if (auto cOp = dyn_cast_or_null<arith::ConstantIndexOp>(v.getDefiningOp())) {
    return cOp.value();
  }

  // handle other constant-related ops
  // support DimOp for now
  if (auto dimOp = dyn_cast_or_null<memref::DimOp>(v.getDefiningOp())) {
    if (auto maybeIndex = dimOp.getConstantIndex()) {
      if (maybeIndex.hasValue()) {
        return dimOp.source().getType().dyn_cast<ShapedType>().getDimSize(
            maybeIndex.getValue());
      }
    }
  }
  return defaultLit;
}

llvm::SmallVector<int64_t, 4>
mlir::getLiteralsFromConstantLikes(ArrayRef<Value> values, int64_t defaultLit) {
  SmallVector<int64_t, 4> res;
  for (auto v : values) {
    res.push_back(getLiteralFromConstantLike(v, defaultLit));
  }
  return res;
}

SmallVector<int64_t, 4> mlir::createOneHot(unsigned size, unsigned offset,
                                           int64_t val) {
  assert(offset < size && "offset should be smaller than size");
  SmallVector<int64_t, 4> res(size, 0);
  res[offset] = val;
  return res;
}

SmallVector<unsigned, 4> mlir::getAllIndicesForNonZeros(ArrayRef<int64_t> vec) {
  SmallVector<unsigned, 4> res;
  for (unsigned i = 0; i < vec.size(); ++i) {
    if (vec[i] != 0) {
      res.push_back(i);
    }
  }
  return res;
}

bool mlir::isConstantIndex(Value value, int64_t lit) {
  if (auto def = value.getDefiningOp<arith::ConstantIndexOp>()) {
    return def.value() == lit;
  }
  return false;
}

bool mlir::isZeroAttribute(Attribute value) {
  if (auto intValue = value.dyn_cast<IntegerAttr>())
    return intValue.getValue().isZero();
  if (auto fpValue = value.dyn_cast<FloatAttr>())
    return fpValue.getValue().isZero();
  if (auto splatValue = value.dyn_cast<SplatElementsAttr>())
    return isZeroAttribute(splatValue.getSplatValue<Attribute>());
  if (auto elementsValue = value.dyn_cast<ElementsAttr>())
    return llvm::all_of(elementsValue.getValues<Attribute>(), isZeroAttribute);
  if (auto arrayValue = value.dyn_cast<ArrayAttr>())
    return llvm::all_of(arrayValue.getValue(), isZeroAttribute);
  return false;
}

bool mlir::isMinValueAttribute(Attribute value) {
  if (auto intValue = value.dyn_cast<IntegerAttr>())
    return intValue.getValue().isMinValue();
  if (auto fpValue = value.dyn_cast<FloatAttr>())
    return fpValue.getValue().isInfinity() &&
           fpValue.getValue().isNegative(); // -inf
  if (auto splatValue = value.dyn_cast<SplatElementsAttr>())
    return isMinValueAttribute(splatValue.getSplatValue<Attribute>());
  if (auto elementsValue = value.dyn_cast<ElementsAttr>())
    return llvm::all_of(elementsValue.getValues<Attribute>(),
                        isMinValueAttribute);
  if (auto arrayValue = value.dyn_cast<ArrayAttr>())
    return llvm::all_of(arrayValue.getValue(), isMinValueAttribute);
  return false;
}

bool mlir::isSplatValue(DenseIntElementsAttr attr, int64_t value) {
  if (!attr) {
    return false;
  }
  if (!attr.isSplat()) {
    return false;
  }
  int64_t start_val = attr.getSplatValue<IntegerAttr>().getInt();
  return start_val == value;
}

// Returns true if the given `attr` is a splat value as the given `value`.
bool mlir::isSplatValue(DenseFPElementsAttr attr, double value) {
  if (!attr)
    return false;
  return attr.isSplat() &&
         attr.getSplatValue<FloatAttr>().getValueAsDouble() == value;
}

bool mlir::isSplatCloseToValue(DenseFPElementsAttr attr, double value,
                               double EPSILON /*=0.00001*/) {
  if (!attr)
    return false;
  if (!attr.isSplat())
    return false;
  double x = attr.getSplatValue<FloatAttr>().getValueAsDouble() - value;
  if ((x >= -EPSILON) && (x <= EPSILON))
    return true;
  return false;
}

std::string mlir::getAttrPlaceholderName(StringRef name) {
  return "__placeholder__" + name.str();
}

void mlir::removeAttrPlaceholders(mlir::Operation *op,
                                  ArrayRef<StringRef> OrgNames) {

  for (const auto &name : OrgNames) {
    auto placeholder = getAttrPlaceholderName(name);
    auto attr = op->getAttr(placeholder);
    if (attr == nullptr) {
      continue;
    }

    op->setAttr(name, attr);
    op->removeAttr(placeholder);
  }
}

mlir::FuncOp mlir::GetFuncOp(mlir::CallOp callOp) {
  Operation *op = callOp.getOperation();
  CallOpInterface call = dyn_cast<CallOpInterface>(op);
  Operation *defOp = call.resolveCallable();
  if (auto funcOp = dyn_cast<mlir::FuncOp>(defOp)) {
    return funcOp;
  }
  // if not a FuncOp, return a null
  return FuncOp();
}

bool mlir::HasAnyOfAttrs(llvm::ArrayRef<mlir::NamedAttribute> attrs,
                         llvm::ArrayRef<llvm::StringRef> filterAttrs) {
  if (filterAttrs.empty())
    return true;

  if (attrs.empty()) {
    return false;
  }

  SmallVector<NamedAttribute, 8> filteredAttrs(
      llvm::make_filter_range(attrs, [&](NamedAttribute attr) {
        return llvm::is_contained(filterAttrs, attr.getName().getValue());
      }));

  return !filteredAttrs.empty();
}

void mlir::AddAttrs(mlir::Operation *op,
                    llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  for (auto attr : attrs) {
    // override if there is any with the same name
    op->setAttr(attr.getName(), attr.getValue());
  }
}

Optional<unsigned> mlir::FindOperandIndex(mlir::Operation *op,
                                          mlir::Value val) {
  if (op == nullptr) {
    return None;
  }

  auto num_operand = op->getNumOperands();
  for (unsigned i = 0; i < num_operand; ++i) {
    if (val == op->getOperand(i)) {
      return i;
    }
  }
  return None;
}

Optional<unsigned> mlir::FindResultIndex(mlir::Operation *op, mlir::Value val) {
  if (op == nullptr || val.getDefiningOp() != op) {
    return None;
  }

  auto num_result = op->getNumResults();
  for (unsigned i = 0; i < num_result; ++i) {
    if (val == op->getResult(i)) {
      return i;
    }
  }
  return None;
}

void mlir::getValuesFromDenseIntElementsAttr(
    DenseIntElementsAttr attr, SmallVector<int64_t> &arrayValues) {
  for (auto it = attr.begin(); it != attr.end(); it++) {
    arrayValues.push_back((*it).getSExtValue());
  }
}

SmallVector<Value, 4>
mlir::GetInputsOfCluster(const llvm::SmallVector<Operation *, 8> &cluster) {
  SmallVector<Value, 4> inputs;
  SmallDenseSet<Value> input_set;
  SmallDenseSet<Operation *> op_set;
  for (Operation *op : cluster) {
    bool inserted = op_set.insert(op).second;
    (void)inserted;
    assert(inserted && "cluster contains duplicate operations");
  }

  for (Operation *op : cluster) {
    for (Value operand : op->getOperands()) {
      Operation *operand_op = operand.getDefiningOp();
      if (op_set.find(operand_op) != op_set.end()) {
        // skip if defining op is in the cluster
        continue;
      }
      if (input_set.insert(operand).second) {
        inputs.push_back(operand);
      }
    }
  }
  return inputs;
}

SmallVector<Value, 4>
mlir::GetOutputsOfCluster(const llvm::SmallVector<Operation *, 8> &cluster) {
  SmallVector<Value, 4> outputs;
  SmallDenseSet<Operation *> op_set;
  for (Operation *op : cluster) {
    bool inserted = op_set.insert(op).second;
    (void)inserted;
    assert(inserted && "cluster contains duplicate operations");
  }

  for (Operation *op : cluster) {
    for (Value result : op->getResults()) {
      bool has_external_user =
          llvm::any_of(result.getUses(), [&](OpOperand &use) {
            return !op_set.count(use.getOwner());
          });
      if (has_external_user) {
        outputs.push_back(result);
      }
    }
  }
  return outputs;
}

bool mlir::IsMemrefTrivial(mlir::Value memref,
                           llvm::ArrayRef<mlir::Operation *> filters) {
  SmallPtrSet<mlir::Operation *, 4> op_sets(filters.begin(), filters.end());

  if (!memref.getDefiningOp<memref::AllocOp>()) {
    return false;
  }

  for (Operation *user : memref.getUsers()) {
    if (!op_sets.contains(user) || !isa<memref::DeallocOp>(user)) {
      return false;
    }
  }
  return true;
}

int mlir::UserCount(Value val) {
  SmallDenseSet<Operation *> count;
  for (auto user : val.getUsers()) {
    if (!count.contains(user)) {
      count.insert(user);
    }
  }
  return count.size();
}
