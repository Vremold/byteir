//===- kernels.cc ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "kernels.h"
#include "brt/backends/common.h"
#include "brt/backends/cuda/device/common/util.h"
#include "brt/core/common/common.h"
#include "brt/core/common/status.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/framework/kernel_registry.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/framework/op_kernel.h"
#include <cuda_runtime.h>
#include <iostream>

using namespace brt;
using namespace brt::cuda;
using namespace brt::common;
using namespace brt::ir;

namespace {
template <typename T> class CustomAdd final : public OpKernel {
public:
  explicit CustomAdd(const OpKernelInfo &info)
      : OpKernel(info, true, false, true, false) {}

  common::Status RunImpl(const ExecutionContext &) override;

  common::Status ProloguePerSession() override;

  common::Status ProloguePerFrame(const ExecutionContext &) override;
};

template <typename T>
common::Status CustomAdd<T>::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  Shape shape = accessor.GetArgShape(0);
  int64_t n = accessor.GetNumElementsOfShape(shape);

  auto p = MakeCUDAGridAndBlock(n);
  size_t dyn_shared_size = 0;

  std::vector<void *> args;
  args.push_back(&p.first);         // grid
  args.push_back(&p.second);        // block
  args.push_back(&dyn_shared_size); // dyn_shared_size

  auto num_arg = accessor.GetNumArgs();
  std::vector<AsyncValueRef> ptrs(num_arg);
  for (unsigned int i = 0; i < num_arg; ++i) {
    ptrs[i] = accessor.GetArgAsyncValueRef(i);
    args.push_back(&ptrs[i]);
  }

  args.push_back(&n); // n

  ctx.work_queue->AddTask(0, (void *)external_kernels::add_kernel<T>,
                          args.data());

  return Status::OK();
}

template <typename T> common::Status CustomAdd<T>::ProloguePerSession() {
  std::cout << "this is CustomizeAddOp ProloguePerSession" << std::endl;
  return Status::OK();
}

template <typename T>
common::Status CustomAdd<T>::ProloguePerFrame(const ExecutionContext &) {
  std::cout << "this is CustomizeAddOp ProloguePerFrame" << std::endl;
  return Status::OK();
}

// statcially register all CPU OpKernels
BRT_STATIC_KERNEL_REGISTRATION(
    DeviceKind::CUDA, ProviderType::BRT, [](KernelRegistry *registry) {
      registry->Register(
          "CustomAddOp",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            auto kernel = std::shared_ptr<OpKernel>(new CustomAdd<float>(info));
            return kernel;
          });
    });
} // namespace
