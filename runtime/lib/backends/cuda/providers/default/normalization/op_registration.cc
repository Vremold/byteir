//===- op_registration.cc -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/backends/cuda/providers/default/normalization/op_registration.h"
#include "brt/core/framework/kernel_registry.h"

#include "./batch_norm_grad.h"
#include "./batch_norm_training.h"
#include <cuda_fp16.h>

namespace brt {
namespace cuda {

void RegisterNormalizationOps(KernelRegistry *registry) {
  registry->Register(
      "BatchNormTrainingOpf16f32f32f16f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(
            new cuda::BatchNormTraining<__half>(info));
        return kernel;
      });
  registry->Register(
      "BatchNormTrainingOpf32f32f32f32f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::BatchNormTraining<float>(info));
        return kernel;
      });
  registry->Register(
      "BatchNormTrainingOpf16f32f32f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(
            new cuda::BatchNormTrainingNoMeanVar<__half>(info));
        return kernel;
      });
  registry->Register(
      "BatchNormTrainingOpf32f32f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(
            new cuda::BatchNormTrainingNoMeanVar<float>(info));
        return kernel;
      });

  registry->Register(
      "BatchNormGradOpf16f32f16f16f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::BatchNormGrad<__half>(info));
        return kernel;
      });
  registry->Register(
      "BatchNormGradOpf32f32f32f32f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::BatchNormGrad<float>(info));
        return kernel;
      });
}

} // namespace cuda
} // namespace brt
