//===- copy.h -------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/framework/op_kernel.h"

namespace brt {
namespace cuda {

class CopyOpKernel final : public OpKernel {
public:
  // task_type 1 as H2D
  //           2 as D2H
  CopyOpKernel(const OpKernelInfo &, int task_type);
  ~CopyOpKernel();
  common::Status RunImpl(const ExecutionContext &) override;

private:
  int task_type = 0;
  size_t dst_id = 0;
  size_t src_id = 0;
  size_t byte_size = 0;
};

} // namespace cuda
} // namespace brt
