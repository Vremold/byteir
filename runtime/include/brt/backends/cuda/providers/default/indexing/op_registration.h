//===- op_registration.h --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once
namespace brt {

class KernelRegistry;

namespace cuda {

void RegisterIndexingOps(KernelRegistry *registry);

} // namespace cuda
} // namespace brt
