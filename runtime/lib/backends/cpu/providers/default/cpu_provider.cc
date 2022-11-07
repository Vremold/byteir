//===- cpu_provider.cc ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/backends/cpu/providers/default/cpu_provider.h"

#include "brt/backends/common.h"
#include "brt/backends/cpu/providers/default/math/elementwise_ops.h" // TODO move to another header
#include "brt/core/framework/execution_provider.h"
#include "brt/core/session/session.h"

#include "half/half.hpp"

#include "./custom_call/tf_equal.h"
#include "./custom_call/tf_where.h"
#include "./llvm/jit.h"
#include "./shape/shape_compute.h"
#include "./tensor_generate/fill.h"
#include "./typecvt/typecvt.h"

#include <memory>

using namespace brt;
using namespace brt::common;

namespace brt {

namespace {

// statcially register all CPU OpKernels
BRT_STATIC_KERNEL_REGISTRATION(
    DeviceKind::CPU, ProviderType::BRT, [](KernelRegistry *registry) {
      registry->Register(
          "AddOpf32f32f32",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            auto kernel = std::shared_ptr<OpKernel>(new cpu::Add<float>(info));
            return kernel;
          });
      registry->Register(
          "Typecvti64i32",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            auto kernel = std::shared_ptr<OpKernel>(
                new cpu::Typecvt<DTypeEnum::Int64, DTypeEnum::Int32>(info));
            return kernel;
          });
      registry->Register(
          "Typecvtf32f16",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            auto kernel = std::shared_ptr<OpKernel>(
                new cpu::Typecvt<DTypeEnum::Float32, DTypeEnum::Float16>(info));
            return kernel;
          });
      registry->Register(
          "Typecvtf16f32",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            auto kernel = std::shared_ptr<OpKernel>(
                new cpu::Typecvt<DTypeEnum::Float16, DTypeEnum::Float32>(info));
            return kernel;
          });
      registry->Register(
          "LLVMJITOp",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            return std::make_shared<cpu::LLVMJITOpKernel>(info);
          });
      registry->Register(
          "ComputeShapeOp",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            return std::make_shared<cpu::ShapeCompute>(info);
          });
      registry->Register(
          "FillOp",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
            return std::make_shared<cpu::Fill>(info);
          });
      registry->Register(
          "tf.Where",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            return std::make_shared<cpu::TFWhere>(info);
          });
      registry->Register(
          "tf.Equal",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            return std::make_shared<cpu::TFEqual>(info);
          });
      RegisterCommonBuiltinOps(registry);
    });

} // namespace

CPUExecutionProvider::CPUExecutionProvider(const std::string &name)
    : ExecutionProvider(DeviceKind::CPU, name) {}

common::Status NaiveCPUExecutionProviderFactory(Session *session) {
  // create a CPU provider
  auto cpu_provider = std::make_unique<CPUExecutionProvider>();

  // give ownership to the session
  return session->AddExecutionProvider(std::move(cpu_provider));
}

} // namespace brt
