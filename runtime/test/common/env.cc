//===- env.cc -------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/test/common/env.h"
#include "brt/core/common/logging/sinks/cerr_sink.h"
#include <memory>

namespace brt {
namespace test {

Env *Env::GetInstance() {
  static Env instance;
  return &instance;
}

Env::Env() {
  name_ = "TestEnvLoggingManager";

  logging_manager_ = std::make_unique<brt::logging::LoggingManager>(
      std::unique_ptr<brt::logging::ISink>{
          new brt::logging::CErrSink{}} /*sink*/,
      brt::logging::Severity::kWARNING, false,
      brt::logging::LoggingManager::InstanceType::Default, &name_);
}

} // namespace test
} // namespace brt
