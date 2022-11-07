//===- shape_compute.h ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/framework/op_kernel.h"

namespace brt {
namespace cpu {

class ShapeCompute final : public OpKernel {
public:
  explicit ShapeCompute(const OpKernelInfo &);

  ~ShapeCompute();

  common::Status RunImpl(const ExecutionContext &ctx) override;

private:
  void *symbol;
};

} // namespace cpu
} // namespace brt
