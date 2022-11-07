//===- op_registration.cc -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "./transpose.h"

#include "brt/backends/cuda/providers/default/tensor_manipulate/op_registration.h"
#include "brt/core/framework/kernel_registry.h"
#include <cuda_fp16.h>

namespace brt {
namespace cuda {

void RegisterTensorManipulateOps(KernelRegistry *registry) {
  registry->Register(
      "TransposeOpf16f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<Transpose<__half>>(info);
      });

  registry->Register(
      "TransposeOpf32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<Transpose<float>>(info);
      });
}

} // namespace cuda
} // namespace brt
