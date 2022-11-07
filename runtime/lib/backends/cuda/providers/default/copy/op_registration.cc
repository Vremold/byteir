//===- op_registration.cc -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "./copy.h"

#include "brt/backends/cuda/providers/default/copy/op_registration.h"
#include "brt/core/framework/kernel_registry.h"

namespace brt {
namespace cuda {

void RegisterCopyOps(KernelRegistry *registry) {
  registry->Register(
      "cpu2cuda",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<CopyOpKernel>(info, 1 /*H2D*/);
      });

  registry->Register(
      "cuda2cpu",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<CopyOpKernel>(info, 2 /*D2H*/);
      });
}

} // namespace cuda
} // namespace brt
