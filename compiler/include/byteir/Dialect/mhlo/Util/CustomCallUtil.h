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

constexpr llvm::StringRef getSoftmaxName() {
  return CUSTOM_CALL_NAME_PREFIX "softmax";
}

constexpr llvm::StringRef getLogSoftmaxName() {
  return CUSTOM_CALL_NAME_PREFIX "log_softmax";
}

constexpr llvm::StringRef getGeLUName() {
  return CUSTOM_CALL_NAME_PREFIX "gelu";
}

constexpr llvm::StringRef getErfName() { return CUSTOM_CALL_NAME_PREFIX "erf"; }

constexpr llvm::StringRef getTopKName() {
  return CUSTOM_CALL_NAME_PREFIX "top_k";
}

constexpr llvm::StringRef getArgMaxName() {
  return CUSTOM_CALL_NAME_PREFIX "arg_max";
}

constexpr llvm::StringRef getArgMinName() {
  return CUSTOM_CALL_NAME_PREFIX "arg_min";
}

constexpr llvm::StringRef getLayerNormName() {
  return CUSTOM_CALL_NAME_PREFIX "layer_norm";
}

constexpr llvm::StringRef getL2NormName() {
  return CUSTOM_CALL_NAME_PREFIX "l2_norm";
}

constexpr llvm::StringRef getOneHotName() {
  return CUSTOM_CALL_NAME_PREFIX "one_hot";
}

constexpr llvm::StringRef getAddNName() {
  return CUSTOM_CALL_NAME_PREFIX "addn";
}

constexpr llvm::StringRef getDynamicPartitionName() {
  return TF_NAME_PREFIX "DynamicPartition";
}

constexpr llvm::StringRef getDynamicStitchName() {
  return TF_NAME_PREFIX "DynamicStitch";
}

constexpr llvm::StringRef getDynamicMaskStitchName() {
  return TF_NAME_PREFIX "DynamicMaskStitch";
}

constexpr llvm::StringRef getWhereName() { return TF_NAME_PREFIX "Where"; }

} // namespace mlir

#undef TF_NAME_PREFIX
#undef CUSTOM_CALL_NAME_PREFIX

#endif // BYTEIR_DIALECT_MHLO_UTIL_CUSTOMCALLUTIL_H