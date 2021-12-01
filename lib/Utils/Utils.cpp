//===- Utils.cpp ----------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;
using namespace mlir;

bool mlir::isConstantIndex(Value value, int64_t lit) {
  if (auto def = value.getDefiningOp<ConstantIndexOp>())
    return def.getValue() == lit;
  return false;
}

bool mlir::isZeroAttribute(Attribute value) {
  if (auto intValue = value.dyn_cast<IntegerAttr>())
    return intValue.getValue().isNullValue();
  if (auto fpValue = value.dyn_cast<FloatAttr>())
    return fpValue.getValue().isZero();
  if (auto splatValue = value.dyn_cast<SplatElementsAttr>())
    return isZeroAttribute(splatValue.getSplatValue());
  if (auto elementsValue = value.dyn_cast<ElementsAttr>())
    return llvm::all_of(elementsValue.getValues<Attribute>(), isZeroAttribute);
  if (auto arrayValue = value.dyn_cast<ArrayAttr>())
    return llvm::all_of(arrayValue.getValue(), isZeroAttribute);
  return false;
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
        return llvm::is_contained(filterAttrs, attr.first.strref());
      }));

  return !filteredAttrs.empty();
}

void mlir::AddAttrs(mlir::Operation *op,
                    llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  for (auto attr : attrs) {
    // override if there is any with the same name
    op->setAttr(attr.first, attr.second);
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