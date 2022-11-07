// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// ===========================================================================
// Modifications Copyright (c) ByteDance.

#pragma once

#include "brt/core/common/logging/logging.h"
#include "brt/core/common/status.h"
#include "brt/core/framework/allocator.h"
#include "brt/core/framework/kernel_registry.h"

#include <string>
#include <vector>

namespace brt {

/**
 * ExecutionProvider organizes a set of kernels
 * with a specific set of allocators.
 *
 * Multiple ExecutionProvider's can work together for a given device.
 *
 * All run in ExecutionProvider is asynchronous,
 * synchronous run is considered as a special case.
 */

class ExecutionProvider {
protected:
  ExecutionProvider(const std::string &deviceKind, const std::string &name)
      : deviceKind_{deviceKind}, name_{name} {
    kernel_registry_ = std::make_unique<KernelRegistry>();
    RegisterKernels(deviceKind_, name_, kernel_registry_.get());
  }

public:
  virtual ~ExecutionProvider() = default;

  /**
   * Set KernelRegistry
   * This API can be used to set a KernelRegistry externally and dynamically
   */
  void SetKernelRegistry(std::unique_ptr<KernelRegistry> kernels) {
    kernel_registry_ = std::move(kernels);
  }

  /**
   * Return KernelRegistry for this ExecutionProvider
   */
  KernelRegistry *GetKernelRegistry() const { return kernel_registry_.get(); }

  /**
   * Get execution provider's configuration options.
   */
  // virtual ProviderOptions GetProviderOptions() const { return {}; }

  const std::string &DeviceKind() const { return deviceKind_; }

  const std::string &Name() const { return name_; }

  void SetLogger(const logging::Logger *logger) { logger_ = logger; }

  const logging::Logger *GetLogger() const { return logger_; }

  /**
   * Load kernels from given dynamic library
   */
  static common::Status
  StaticRegisterKernelsFromDynlib(const std::string &path);

  // TODO: implement dynamic register
  common::Status RegisterKernelsFromDynlib(const std::string &path);

protected:
  // a string to check device kind
  const std::string deviceKind_;

  // a string to check name
  const std::string name_;

  // MemoryInfoSet mem_info_set_;  // to ensure only allocators with unique
  // memory info are registered in the provider. It will be set when this object
  // is registered to a session
  const logging::Logger *logger_ = nullptr;

  std::unique_ptr<KernelRegistry> kernel_registry_;

private:
  // disallow copy
  ExecutionProvider(const ExecutionProvider &) = delete;

  // disable assignment
  ExecutionProvider &operator=(const ExecutionProvider &) = delete;

  // disallow move
  ExecutionProvider(ExecutionProvider &&) = delete;
  ExecutionProvider &operator=(ExecutionProvider &&) = delete;
};
} // namespace brt
