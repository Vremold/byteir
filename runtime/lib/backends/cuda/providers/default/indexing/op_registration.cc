//===- op_registration.cc -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "./index_put.h"
#include "./index_select.h"

#include "brt/backends/cuda/providers/default/indexing/op_registration.h"
#include "brt/core/framework/kernel_registry.h"

namespace brt {
namespace cuda {

void RegisterIndexingOps(KernelRegistry *registry) {
  registry->Register(
      "IndexSelectOpf32ui32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<IndexSelect<float>>(info);
      });

  registry->Register(
      "IndexPutOpf32i64f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<IndexPut<float>>(info);
      });
}

} // namespace cuda
} // namespace brt
