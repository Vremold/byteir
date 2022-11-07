//===- fill.cc ------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//
#include "./fill.h"
#include "brt/core/framework/op_accessor.h"

namespace brt {
namespace cpu {

common::Status Fill::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  DTypeEnum dtype = accessor.GetArgDTypeEnum(0);
  void *p = accessor.GetArgAsyncValueRef(0);
  size_t length = accessor.GetNumElementsOfShape(accessor.GetArgShape(0));
  switch (dtype) {
  case DTypeEnum::StringView: {
    // TODO: take the ownership of the underlying data of the
    // string_view which belongs to IRHandle
    std::fill_n(reinterpret_cast<StringView *>(p), length,
                accessor.GetAttrAsSplatValue<StringView>("value"));
    return common::Status::OK();
  }
  default:
    return common::Status(common::StatusCategory::BRT,
                          common::StatusCode::NOT_IMPLEMENTED,
                          "not supported dtype");
  }
}

} // namespace cpu
} // namespace brt
