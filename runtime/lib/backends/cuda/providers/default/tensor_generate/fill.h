//===- fill.h -------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/framework/dtype.h"
#include "brt/core/framework/op_kernel.h"

namespace brt {
namespace cuda {
// TODO: Strictly speaking FillOpKernel should be a per session constant
// rather than per frame.
//
// TODO: Currently we only have a simple memory assignment strategy without any
// memory reusing. But when we try to do that, we need to mark the memory buffer
// filled with given constant as non-reusable, since we only initialize and
// write to the buffer before the first run for each frame. And maybe managing
// buffer by FillOpKernel itself is better than planning it in ExecutionPlan, if
// we decide to change the behavior of FillOp from per-frame to per-session,
class FillOpKernel final : public OpKernel {
public:
  explicit FillOpKernel(const OpKernelInfo &info);
  common::Status RunImpl(const ExecutionContext &) override;
  common::Status ProloguePerFrame(const ExecutionContext &) override;
  common::Status EpiloguePerFrame(const ExecutionContext &) override;
};

} // namespace cuda
} // namespace brt
