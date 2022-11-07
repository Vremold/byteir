//===- util.cc ------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/core/ir/util.h"
#include "brt/core/common/exceptions.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeUtilities.h"

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace mlir;

namespace brt {
namespace ir {

// Get total bytes of a memref
uint64_t GetStaticBytes(mlir::MemRefType memref) {
  return memref.getNumElements() * GetElementTypeByte(memref);
}

// Get total bytes of a value if it is a memref
// Return None if a value is not a memref
llvm::Optional<uint64_t> GetStaticBytes(mlir::Value val) {
  if (auto memref = val.getType().dyn_cast<mlir::MemRefType>()) {
    return GetStaticBytes(memref);
  }
  return llvm::None;
}

// Get static shape in IR, negative value for unknown
llvm::Optional<llvm::ArrayRef<int64_t>> GetStaticShape(mlir::Value val) {
  if (auto memref = val.getType().dyn_cast<mlir::MemRefType>()) {
    return memref.getShape();
  }
  return llvm::None;
}

llvm::Optional<uint64_t> GetElementTypeByte(mlir::Value val) {
  if (auto memref = val.getType().dyn_cast<mlir::MemRefType>()) {
    return GetElementTypeByte(memref);
  }
  return llvm::None;
}

llvm::Optional<size_t> GetRank(mlir::Value val) {
  if (auto memref = val.getType().dyn_cast<mlir::MemRefType>()) {
    return static_cast<size_t>(memref.getRank());
  }
  return llvm::None;
}

// Get space in IR, empty value for unknown
std::string GetSpace(mlir::MemRefType memref) {
  if (auto str_attr = memref.getMemorySpace().dyn_cast_or_null<StringAttr>()) {
    return str_attr.str();
  }
  return "";
}

llvm::Optional<std::string> GetSpace(mlir::Value val) {
  if (auto memref = val.getType().dyn_cast<mlir::MemRefType>()) {
    if (auto str_attr =
            memref.getMemorySpace().dyn_cast_or_null<StringAttr>()) {
      return str_attr.str();
    }
    return std::string();
  }
  return llvm::None;
}

DTypeEnum GetElementDTypeEnum(mlir::Value val) {
  Type elementType;
  if (auto memref = val.getType().dyn_cast<mlir::MemRefType>()) {
    elementType = memref.getElementType();
  } else {
    return DTypeEnum::Invalid;
  }
  return ConvertMLIRTypeToDType(elementType);
}

llvm::Optional<int64_t> LinearizedStaticShape(llvm::ArrayRef<int64_t> shape) {
  int64_t res = 1;
  for (auto d : shape) {
    if (d <= 0) {
      return llvm::None;
    }
    res *= d;
  }
  return res;
}

int64_t GetIntegerAttrValue(mlir::Attribute attr) {
  mlir::IntegerAttr integerAttr = attr.dyn_cast<mlir::IntegerAttr>();
  BRT_ENFORCE(integerAttr, "must be Integer Attribute");
  return integerAttr.getValue().getSExtValue();
}

bool IsComptaibleShapeOf(const std::vector<int64_t> &shape, mlir::Value value) {
  if (auto memref = value.getType().dyn_cast<mlir::MemRefType>()) {
    return verifyCompatibleShape(shape, memref.getShape()).succeeded();
  }
  return false;
}

bool IsByreStringType(mlir::Type type) {
  // TODO: introduce byre.string instead of ace.string
  return type.isa<ace::StringType>();
}

// TODO: introduce byre.string
mlir::Type CreateByreStringType(mlir::MLIRContext *context) {
  // TODO: introduce byre.string instead of ace.string
  return ace::StringType::get(context);
}

} // namespace ir
} // namespace brt
