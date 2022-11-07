//===- util.h -------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/common/common.h"
#include "brt/core/common/status.h"
#include "brt/core/framework/dtype.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"

/**
 * This file holds utility functions for IR
 */

namespace brt {
namespace ir {

bool IsByreStringType(mlir::Type type);

mlir::Type CreateByreStringType(mlir::MLIRContext *context);

inline DTypeEnum ConvertMLIRTypeToDType(mlir::Type elementType) {
  if (elementType.isF32()) {
    return DTypeEnum::Float32;
  } else if (elementType.isSignlessInteger(32)) {
    return DTypeEnum::Int32;
  } else if (elementType.isSignlessInteger(64)) {
    return DTypeEnum::Int64;
  } else if (elementType.isUnsignedInteger(8)) {
    return DTypeEnum::UInt8;
  } else if (elementType.isUnsignedInteger(32)) {
    return DTypeEnum::UInt32;
  } else if (elementType.isF16()) {
    return DTypeEnum::Float16;
  } else if (elementType.isBF16()) {
    return DTypeEnum::BFloat16;
  } else if (elementType.isF64()) {
    return DTypeEnum::Float64;
  } else if (elementType.isSignlessInteger(1)) {
    return DTypeEnum::Bool;
  } else if (IsByreStringType(elementType)) {
    return DTypeEnum::StringView;
  }
  return DTypeEnum::Unsupported;
}

inline mlir::Type ConvertDTypeToMLIRType(DTypeEnum dtype,
                                         mlir::MLIRContext *context) {
  using SignednessSemantics = mlir::IntegerType::SignednessSemantics;
  switch (dtype) {
  case DTypeEnum::Float32:
    return mlir::FloatType::getF32(context);
  case DTypeEnum::Int32:
    return mlir::IntegerType::get(context, 32, SignednessSemantics::Signless);
  case DTypeEnum::Int64:
    return mlir::IntegerType::get(context, 64, SignednessSemantics::Signless);
  case DTypeEnum::UInt8:
    return mlir::IntegerType::get(context, 8, SignednessSemantics::Unsigned);
  case DTypeEnum::UInt32:
    return mlir::IntegerType::get(context, 32, SignednessSemantics::Unsigned);
  case DTypeEnum::Float16:
    return mlir::FloatType::getF16(context);
  case DTypeEnum::BFloat16:
    return mlir::FloatType::getBF16(context);
  case DTypeEnum::Float64:
    return mlir::FloatType::getF64(context);
  case DTypeEnum::Bool:
    return mlir::IntegerType::get(context, 1, SignednessSemantics::Signless);
  case DTypeEnum::StringView:
    return CreateByreStringType(context);
  default:
    return nullptr;
  }
}

// TODO support symbolic later

// Get total bytes of a memref
uint64_t GetStaticBytes(mlir::MemRefType memref);

// Get total bytes of a value if it is a memref
// Return None if a value is not a memref
llvm::Optional<uint64_t> GetStaticBytes(mlir::Value val);

// Get element in byte of a memref
inline unsigned int GetElementTypeByte(mlir::MemRefType memref) {
  // make sure round up to 1 byte
  auto elementType = memref.getElementType();
  if (elementType.isIntOrIndexOrFloat()) {
    return (elementType.getIntOrFloatBitWidth() + 7) >> 3;
  }
  auto dtype = ConvertMLIRTypeToDType(elementType);
  switch (dtype) {
#define Case(D)                                                                \
  case DTypeEnum::D: {                                                         \
    return sizeof(DTypeTraits<DTypeEnum::D>::type_t);                          \
  }
    Case(StringView);
#undef Case
  default: {
    BRT_THROW("invalid element type");
  }
  }
}

// Get element in byte of a value if it is a memref
// Return None if a value is not a memref
llvm::Optional<uint64_t> GetElementTypeByte(mlir::Value val);

// Get static shape in IR, negative value for unknown
llvm::Optional<llvm::ArrayRef<int64_t>> GetStaticShape(mlir::Value val);

llvm::Optional<size_t> GetRank(mlir::Value val);

// Get space in IR, empty value for unknown
std::string GetSpace(mlir::MemRefType memref);
llvm::Optional<std::string> GetSpace(mlir::Value val);

// Get dtype of given IR Value
DTypeEnum GetElementDTypeEnum(mlir::Value val);

template <typename T> inline bool IsElementType(mlir::Value val) {
  if (auto memref = val.getType().dyn_cast<mlir::MemRefType>()) {
    return memref.getElementType().isa<T>();
  }
  return false;
}

llvm::Optional<int64_t> LinearizedStaticShape(llvm::ArrayRef<int64_t> shape);

// Get IntegerAttr's value
int64_t GetIntegerAttrValue(mlir::Attribute attr);

// return whether \p shape is compatible with the shape of \p value
bool IsComptaibleShapeOf(const std::vector<int64_t> &shape, mlir::Value value);

} // namespace ir
} // namespace brt
