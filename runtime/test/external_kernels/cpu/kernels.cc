//===- kernels.cc ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/backends/common.h"
#include "brt/core/common/common.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/framework/kernel_registry.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/framework/op_kernel.h"
#include <iostream>

using namespace brt;

namespace {
// This is a test Op
// TODO remove it later
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
  std::cout << "this is CustomizeAddOp Run" << std::endl;
  auto num_arg = accessor.GetNumArgs();
  for (unsigned int i = 0; i < num_arg; ++i) {
    AsyncValueRef value = accessor.GetArgAsyncValueRef(i);
    // try access it
    T *ptr = static_cast<T *>(value);
    ptr[0] = (T)1;
  }
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

class GroupAllocationHookChecker : public OpKernel {
public:
  explicit GroupAllocationHookChecker(const OpKernelInfo &info)
      : OpKernel(info, false, false, false, false) {
    OpAccessor accessor(info_);
    base = accessor.GetAttrAsInt("base");
  }

  common::Status RunImpl(const ExecutionContext &) override {
    return Status::OK();
  }

  common::Status GetGroupAllocationHook(
      std::unique_ptr<GroupAllocationHook> *group_allocation_hook) override {
    size_t nr_args = GetOpArgNum(info_);
    auto alloc_f = [=] {
      std::vector<AsyncValue> rets;
      rets.reserve(nr_args);
      for (size_t i = 0; i < nr_args; ++i) {
        rets.push_back(reinterpret_cast<AsyncValue>(base + i));
      }
      return rets;
    };
    auto free_f = [](std::vector<AsyncValue>) {
      /* do nothing */
    };
    *group_allocation_hook =
        std::make_unique<OpKernelGroupAllocationHook>(info_, alloc_f, free_f);
    return Status::OK();
  }

private:
  size_t base;
};

// statcially register all CPU OpKernels
BRT_STATIC_KERNEL_REGISTRATION(
    DeviceKind::CPU, ProviderType::BRT, [](KernelRegistry *registry) {
      registry->Register(
          "CustomAddOp",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            auto kernel = std::shared_ptr<OpKernel>(new CustomAdd<float>(info));
            return kernel;
          });
      registry->Register(
          "CheckGroupAllocationHook",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            auto kernel =
                std::shared_ptr<OpKernel>(new GroupAllocationHookChecker(info));
            return kernel;
          });
    });
} // namespace
