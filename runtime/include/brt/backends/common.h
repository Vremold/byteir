//===- common.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

namespace brt {

struct DeviceKind {
  constexpr static char CPU[] = "CPU";
  constexpr static char CUDA[] = "CUDA";
};

struct ProviderType {
  // type for brt builtin provider
  constexpr static char BRT[] = "BRT";
};

} // namespace brt