//===- Utils.cpp ----------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

llvm::Optional<int64_t> mlir::getLiteralFromConstantLike(Value v) {
  if (auto cOp = dyn_cast_or_null<arith::ConstantIndexOp>(v.getDefiningOp())) {
    return cOp.value();
  }

  // handle other constant-related ops
  // support DimOp for now
  if (auto dimOp = dyn_cast_or_null<memref::DimOp>(v.getDefiningOp())) {
    if (auto maybeIndex = dimOp.getConstantIndex()) {
      if (maybeIndex.has_value()) {
        return dimOp.getSource().getType().dyn_cast<ShapedType>().getDimSize(
            maybeIndex.value());
      }
    }
  }

  return llvm::None;
}

int64_t mlir::getLiteralFromConstantLike(Value v, int64_t defaultLit) {
  auto maybeI64 = getLiteralFromConstantLike(v);
  if (maybeI64.has_value())
    return maybeI64.value();
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
  int64_t startVal = attr.getSplatValue<IntegerAttr>().getInt();
  return startVal == value;
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

mlir::func::FuncOp mlir::getFuncOp(func::CallOp callOp) {
  Operation *op = callOp.getOperation();
  CallOpInterface call = dyn_cast<CallOpInterface>(op);
  Operation *defOp = call.resolveCallable();
  if (auto funcOp = dyn_cast<mlir::func::FuncOp>(defOp)) {
    return funcOp;
  }
  // if not a FuncOp, return a null
  return func::FuncOp();
}

bool mlir::hasAnyOfAttrs(llvm::ArrayRef<mlir::NamedAttribute> attrs,
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

void mlir::addAttrs(mlir::Operation *op,
                    llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  for (auto attr : attrs) {
    // override if there is any with the same name
    op->setAttr(attr.getName(), attr.getValue());
  }
}

Optional<unsigned> mlir::findOperandIndex(mlir::Operation *op,
                                          mlir::Value val) {
  if (op == nullptr) {
    return None;
  }

  auto numOperand = op->getNumOperands();
  for (unsigned i = 0; i < numOperand; ++i) {
    if (val == op->getOperand(i)) {
      return i;
    }
  }
  return None;
}

Optional<unsigned> mlir::findResultIndex(mlir::Operation *op, mlir::Value val) {
  if (op == nullptr || val.getDefiningOp() != op) {
    return None;
  }

  auto numResult = op->getNumResults();
  for (unsigned i = 0; i < numResult; ++i) {
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
mlir::getInputsOfCluster(const llvm::SmallVector<Operation *, 8> &cluster) {
  SmallVector<Value, 4> inputs;
  SmallDenseSet<Value> inputSet;
  SmallDenseSet<Operation *> opSet;
  for (Operation *op : cluster) {
    bool inserted = opSet.insert(op).second;
    (void)inserted;
    assert(inserted && "cluster contains duplicate operations");
  }

  for (Operation *op : cluster) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (opSet.find(defOp) != opSet.end()) {
        // skip if defining op is in the cluster
        continue;
      }
      if (inputSet.insert(operand).second) {
        inputs.push_back(operand);
      }
    }
  }
  return inputs;
}

SmallVector<Value, 4>
mlir::getOutputsOfCluster(const llvm::SmallVector<Operation *, 8> &cluster,
                          const llvm::DenseMap<Value, int64_t> *outputStats) {
  SmallVector<Value, 4> outputs;
  SmallDenseSet<Operation *> opSet;
  for (Operation *op : cluster) {
    // Should add all the operations recursively because a value might be used
    // by an operation of an inner region.
    op->walk([&](Operation *innerOp) {
      bool inserted = opSet.insert(innerOp).second;
      (void)inserted;
      assert(inserted && "cluster contains duplicate operations");
    });
  }

  for (Operation *op : cluster) {
    for (Value result : op->getResults()) {
      bool hasExternalUser =
          llvm::any_of(result.getUses(), [&](OpOperand &use) {
            return !opSet.count(use.getOwner());
          });
      if (hasExternalUser) {
        if (outputStats != nullptr && outputStats->count(result)) {
          outputs.insert(outputs.end(), outputStats->find(result)->second,
                         result);
        } else {
          outputs.push_back(result);
        }
      }
    }
  }
  return outputs;
}

bool mlir::isMemrefTrivial(mlir::Value memref,
                           llvm::ArrayRef<mlir::Operation *> filters) {
  SmallPtrSet<mlir::Operation *, 4> opSet(filters.begin(), filters.end());

  if (!memref.getDefiningOp<memref::AllocOp>()) {
    return false;
  }

  for (Operation *user : memref.getUsers()) {
    if (!opSet.contains(user) || !isa<memref::DeallocOp>(user)) {
      return false;
    }
  }
  return true;
}

int mlir::userCount(Value val) {
  SmallDenseSet<Operation *> count;
  for (auto user : val.getUsers()) {
    if (!count.contains(user)) {
      count.insert(user);
    }
  }
  return count.size();
}