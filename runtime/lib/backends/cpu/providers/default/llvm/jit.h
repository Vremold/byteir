//===- jit.h --------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/framework/op_kernel.h"

namespace brt {
namespace cpu {

class LLVMJITOpKernel final : public OpKernel {
public:
  explicit LLVMJITOpKernel(const OpKernelInfo &);

  ~LLVMJITOpKernel();

  common::Status RunImpl(const ExecutionContext &ctx) override;

private:
  void *symbol;
};

} // namespace cpu
} // namespace brt
