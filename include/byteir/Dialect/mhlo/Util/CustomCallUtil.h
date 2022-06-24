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

namespace mlir {

constexpr llvm::StringRef getCustomCallAttrName() { return "byteir_attrs"; }

constexpr llvm::StringRef getNonZeroName() {
  return CUSTOM_CALL_NAME_PREFIX "non_zero";
}

constexpr llvm::StringRef getDynamicPartitionName() {
  return CUSTOM_CALL_NAME_PREFIX "dynamic_partition";
}

constexpr llvm::StringRef getDynamicStitchName() {
  return CUSTOM_CALL_NAME_PREFIX "dynamic_stitch";
}

} // namespace mlir

#undef CUSTOM_CALL_NAME_PREFIX

#endif // BYTEIR_DIALECT_MHLO_UTIL_CUSTOMCALLUTIL_H