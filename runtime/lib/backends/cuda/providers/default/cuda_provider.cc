//===- cuda_provider.cc ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include "brt/backends/common.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
// TODO make the following to another header file
#include "brt/backends/cuda/providers/default/codegen/op_registration.h"
#include "brt/backends/cuda/providers/default/copy/op_registration.h"
#include "brt/backends/cuda/providers/default/ftv3/bert_transformer.h"
#include "brt/backends/cuda/providers/default/ftv4/faster_transformer.h"
#include "brt/backends/cuda/providers/default/indexing/op_registration.h"
#include "brt/backends/cuda/providers/default/math/op_registration.h"
#include "brt/backends/cuda/providers/default/normalization/op_registration.h"
#include "brt/backends/cuda/providers/default/reduction/op_registration.h"
#include "brt/backends/cuda/providers/default/tensor_generate/op_registration.h"
#include "brt/backends/cuda/providers/default/tensor_manipulate/op_registration.h"
#include "brt/core/session/session.h"
#include <memory>

using namespace brt;
using namespace brt::common;

namespace brt {
namespace {

// statcially register all CUDA OpKernels
// TODO: to use MACRO trick to load all kernels
// TODO: to add dynamic suppport.
// clang-format off
BRT_STATIC_KERNEL_REGISTRATION(
    DeviceKind::CUDA, ProviderType::BRT, [](KernelRegistry *registry) {
      cuda::RegisterBertTransformerOp(registry);
      cuda::RegisterCodegenOps(registry);
      cuda::RegisterCopyOps(registry);
      cuda::RegisterFasterTransformerOps(registry);
      cuda::RegisterIndexingOps(registry);
      cuda::RegisterMathOps(registry);
      cuda::RegisterNormalizationOps(registry);
      cuda::RegisterReductionOps(registry);
      cuda::RegisterTensorGenerateOps(registry);
      cuda::RegisterTensorManipulateOps(registry);
      RegisterCommonBuiltinOps(registry);
    });
// clang-format on

} // namespace

CUDAExecutionProvider::CUDAExecutionProvider(const std::string &name)
    : ExecutionProvider(DeviceKind::CUDA, name) {}

common::Status DefaultCUDAExecutionProviderFactory(Session *session,
                                                   int /*device_id*/) {
  // create a CUDA provider
  auto provider = std::make_unique<CUDAExecutionProvider>();

  // give ownership to the session
  return session->AddExecutionProvider(std::move(provider));
}

} // namespace brt
