//===- cuda_provider.h ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/backends/common.h"
#include "brt/core/common/status.h"
#include "brt/core/framework/execution_provider.h"

namespace brt {
class Session;

class CUDAExecutionProvider : public ExecutionProvider {
public:
  explicit CUDAExecutionProvider(const std::string &name = ProviderType::BRT);
};

// TODO add more option later
common::Status DefaultCUDAExecutionProviderFactory(Session *session,
                                                   int device_id = 0);

} // namespace brt
