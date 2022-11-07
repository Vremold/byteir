//===- elementwise_ops.h --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/framework/op_kernel.h"

namespace brt {
namespace cuda {

/**
 * Add Ops
 * This is just an example for OpKernel.
 * All elementwise ops should be generated through macro or generator.
 */
template <typename T> class Add final : public OpKernel {
public:
  explicit Add(const OpKernelInfo &info) : OpKernel(info) {}

  common::Status RunImpl(const ExecutionContext &) override;
};

} // namespace cuda
} // namespace brt
