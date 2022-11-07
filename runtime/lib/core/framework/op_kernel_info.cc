//===- op_kernel_info.cc --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/core/framework/op_kernel_info.h"

#include "brt/core/common/common.h"
#include "byteir/Dialect/Byre/ByreDialect.h"

using namespace brt;
using namespace brt::common;
using namespace mlir;

namespace brt {

namespace {
inline size_t GetTensorIndexFromOpArgIndexImpl(const OpKernelInfo &info,
                                               unsigned int arg_idx) {
  const std::unordered_map<void *, size_t> &arg_to_idx =
      info.GetTensorToIndex();
  byre::ByreOp byre_op = cast<byre::ByreOp>(info.GetOperation());
  auto op_arg = byre_op->getOperand(arg_idx);
  auto found = arg_to_idx.find(op_arg.getAsOpaquePointer());
  BRT_ENFORCE(found != arg_to_idx.end(),
              "at arg_idx " + std::to_string(arg_idx));
  return found->second;
}
} // namespace

size_t GetTensorIndexFromOpArgIndex(const OpKernelInfo &info,
                                    unsigned int arg_idx) {
  return GetTensorIndexFromOpArgIndexImpl(info, arg_idx);
}

size_t GetTensorIndexFromMLIRValue(const OpKernelInfo &info, mlir::Value val) {
  const std::unordered_map<void *, size_t> &arg_to_idx =
      info.GetTensorToIndex();
  auto found = arg_to_idx.find(val.getAsOpaquePointer());
  BRT_ENFORCE(found != arg_to_idx.end());
  return found->second;
}

size_t GetScalarIndexFromMLIRValue(const OpKernelInfo &info, mlir::Value val) {
  const std::unordered_map<void *, size_t> &arg_to_idx =
      info.GetScalarToIndex();
  auto found = arg_to_idx.find(val.getAsOpaquePointer());
  BRT_ENFORCE(found != arg_to_idx.end());
  return found->second;
}

// Get Rank of MLIR Value, of ith argument of OpKernelInfo
size_t GetRankFromOpArgIndex(const OpKernelInfo &info, unsigned int i) {
  auto value = info.GetOperation()->getOperand(i);
  if (auto memref = value.getType().dyn_cast<mlir::MemRefType>()) {
    return static_cast<size_t>(memref.getRank());
  }
  return 0;
}

unsigned int GetOpArgNum(const OpKernelInfo &info) {
  return info.GetOperation()->getNumOperands();
}

unsigned int GetOpResultNum(const OpKernelInfo &info) {
  return info.GetOperation()->getNumResults();
}

mlir::Value GetMLIRValueFromOpArgIndex(const OpKernelInfo &info,
                                       unsigned int i) {
  return info.GetOperation()->getOperand(i);
}

// return nullptr if not a weight
AsyncValueRef GetWeightFromOpArgIndex(const OpKernelInfo &info,
                                      unsigned int arg_idx) {
  size_t tensor_id = GetTensorIndexFromOpArgIndexImpl(info, arg_idx);

  if (tensor_id < info.GetWeights().size()) {
    return info.GetWeights()[tensor_id];
  }
  return nullptr;
}

mlir::Attribute GetMLIRAttributeFromName(const OpKernelInfo &info,
                                         llvm::StringRef name) {
  return info.GetOperation()->getAttr(name);
}

std::string OpKernelInfo::GetByREOpName() const {
  if (auto op = mlir::dyn_cast<byre::ByreOp>(GetOperation())) {
    return op.getCalleeName();
  }
  return "";
}

} // namespace brt
