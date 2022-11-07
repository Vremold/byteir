//===- execution_provider.cc ----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/core/framework/execution_provider.h"

#ifndef _WIN32
#include <dlfcn.h>
#endif

using namespace brt;
using namespace brt::common;

namespace brt {

Status
ExecutionProvider::StaticRegisterKernelsFromDynlib(const std::string &path) {
#ifdef _WIN32
  return Status(BRT, FAIL, "not implmented");
#else
  void *handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (handle) {
    return Status::OK();
  }
  return Status(BRT, FAIL,
                "cannot open dynamic library " + path + "\n" + dlerror());
#endif
}

} // namespace brt
