//===- env.h --------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/common/logging/logging.h"
#include <string>

namespace brt {
namespace test {

// Env create a global
class Env {
public:
  static Env *GetInstance();

  // TODO disable this for minimal build
  logging::LoggingManager *GetLoggingManager() const {
    return logging_manager_.get();
  }

private:
  Env();

  std::string name_;

  std::unique_ptr<brt::logging::LoggingManager> logging_manager_;
};

} // namespace test
} // namespace brt
