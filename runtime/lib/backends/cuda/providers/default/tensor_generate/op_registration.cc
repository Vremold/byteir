//===- op_registration.cc -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "./fill.h"
#include "./rng.h"

#include "brt/backends/cuda/providers/default/tensor_generate/op_registration.h"
#include "brt/core/framework/kernel_registry.h"

namespace brt {
namespace cuda {

void RegisterTensorGenerateOps(KernelRegistry *registry) {
  registry->Register(
      "FillOp",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<FillOpKernel>(info);
      });
  registry->Register(
      "RngUniform",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<RngUniform>(info);
      });
  registry->Register(
      "RngNormal",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<RngNormal>(info);
      });
}

} // namespace cuda
} // namespace brt
