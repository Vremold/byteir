//===- CustomCallUtil.h ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_UTIL_CUSTOMCALLUTIL_H
#define BYTEIR_DIALECT_MHLO_UTIL_CUSTOMCALLUTIL_H

#include "llvm/ADT/StringRef.h"

#define CUSTOM_CALL_NAME_PREFIX "byteir."
#define TF_NAME_PREFIX "tf."

namespace mlir {

constexpr llvm::StringRef getCustomCallAttrName() { return "byteir_attrs"; }

constexpr llvm::StringRef getNonZeroName() {
  return CUSTOM_CALL_NAME_PREFIX "non_zero";
}

constexpr llvm::StringRef getDynamicPartitionName() {
  return TF_NAME_PREFIX "DynamicPartition";
}

constexpr llvm::StringRef getDynamicStitchName() {
  return TF_NAME_PREFIX "DynamicStitch";
}

} // namespace mlir

#undef CUSTOM_CALL_NAME_PREFIX

#endif // BYTEIR_DIALECT_MHLO_UTIL_CUSTOMCALLUTIL_H
