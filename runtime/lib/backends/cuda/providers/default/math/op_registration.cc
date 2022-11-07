//===- op_registration.cc -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/backends/cuda/providers/default/math/op_registration.h"
#include "brt/core/framework/kernel_registry.h"

#include "./batch_matmul.h"
#include "./conv.h"
#include "./conv_backward.h"
#include "./elementwise_ops.h"
#include "./matmul.h"
#include "./pool.h"
#include "./pool_grad.h"

#include <cuda_fp16.h>

namespace brt {
namespace cuda {

void RegisterMathOps(KernelRegistry *registry) {
  registry->Register(
      "AddOpf32f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(new cuda::Add<float>(info));
        return kernel;
      });

  registry->Register(
      "MatmulOpf16f16f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(new cuda::Matmul<__half>(info));
        return kernel;
      });

  registry->Register(
      "MatmulOpf32f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(new cuda::Matmul<float>(info));
        return kernel;
      });

  registry->Register(
      "BatchMatmulOpf32f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::BatchMatmul<float>(info));
        return kernel;
      });

  registry->Register(
      "ConvOpf32f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(new cuda::Conv<float>(info));
        return kernel;
      });

  registry->Register(
      "ConvOpf16f16f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(new cuda::Conv<__half>(info));
        return kernel;
      });

  registry->Register(
      "ConvBackwardDataOpf32f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::ConvBackwardData<float>(info));
        return kernel;
      });

  registry->Register(
      "ConvBackwardDataOpf16f16f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::ConvBackwardData<__half>(info));
        return kernel;
      });

  registry->Register(
      "ConvBackwardFilterOpf32f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(
            new cuda::ConvBackwardFilter<float>(info));
        return kernel;
      });

  registry->Register(
      "ConvBackwardFilterOpf16f16f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(
            new cuda::ConvBackwardFilter<__half>(info));
        return kernel;
      });

  registry->Register(
      "PoolMaxOpf32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(new cuda::PoolMax<float>(info));
        return kernel;
      });

  registry->Register(
      "PoolMaxOpf16f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::PoolMax<__half>(info));
        return kernel;
      });

  registry->Register(
      "PoolMaxGradOpf32f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::PoolMaxGrad<float>(info));
        return kernel;
      });

  registry->Register(
      "PoolMaxGradOpf16f16f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::PoolMaxGrad<__half>(info));
        return kernel;
      });
}

} // namespace cuda
} // namespace brt
