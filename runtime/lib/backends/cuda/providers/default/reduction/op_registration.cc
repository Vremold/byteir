//===- op_registration.cc -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "./reduce_impl.h"

#include "brt/backends/cuda/providers/default/reduction/op_registration.h"
#include "brt/core/framework/kernel_registry.h"
#include <cuda_fp16.h>

namespace brt {
namespace cuda {

void RegisterReductionOps(KernelRegistry *registry) {

  registry->Register(
      "ReduceSumOpf16f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<ReduceSum<__half>>(info);
      });
  registry->Register(
      "ReduceSumOpf32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<ReduceSum<float>>(info);
      });
  registry->Register(
      "ReduceMaxOpf32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<ReduceMax<float>>(info);
      });
  registry->Register(
      "ReduceMaxOpf16f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<ReduceMax<__half>>(info);
      });
}

} // namespace cuda
} // namespace brt
