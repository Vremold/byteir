//===- op_registration.cc -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "./ptx.h"

#include "brt/backends/cuda/providers/default/codegen/op_registration.h"
#include "brt/core/framework/kernel_registry.h"

namespace brt {
namespace cuda {

void RegisterCodegenOps(KernelRegistry *registry) {
  registry->Register(
      "PTXOp",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<PTXOpKernel>(info);
      });
}

} // namespace cuda
} // namespace brt
