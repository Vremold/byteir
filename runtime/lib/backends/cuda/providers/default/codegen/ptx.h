//===- ptx.h --------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/framework/op_kernel.h"

namespace brt {
namespace cuda {

struct PTXImpl;

class PTXOpKernel final : public OpKernel {
public:
  explicit PTXOpKernel(const OpKernelInfo &);

  ~PTXOpKernel();

  common::Status RunImpl(const ExecutionContext &ctx) override;

private:
  std::unique_ptr<PTXImpl> impl_;
};

} // namespace cuda
} // namespace brt
