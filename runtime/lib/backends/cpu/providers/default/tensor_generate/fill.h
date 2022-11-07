//===- tf_equal.h ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/framework/op_kernel_impl_base.h"

namespace brt {
namespace cpu {

class Fill final : public OpKernel {
public:
  explicit Fill(const OpKernelInfo &info) : OpKernel(info) {}

  common::Status RunImpl(const ExecutionContext &ctx) override;
};

} // namespace cpu
} // namespace brt
