//===- ptx.h --------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/common/status.h"
#include "cuda.h"
#include <memory>
#include <string>
#include <vector>

namespace brt {
namespace cuda {

struct PTXCompilerImpl;

class PTXCompiler {
public:
  PTXCompiler(int device_id);

  ~PTXCompiler();

  bool GetCachedFunction(CUfunction &func, const std::string &name);

  void CreateFunctionFromMemory(CUfunction &func, const std::string &name,
                                const std::string &ptx_str);

  // Get or create a function from a file
  // the file will be accessed if a function is not cached
  // if file = "", it will skip the access.
  common::Status GetOrCreateFunction(CUfunction &func, const std::string &name,
                                     const std::string &file = "");

  // Get or create a function from a file
  // the ptx_str will be accessed if a function is not cached
  // if ptx_str = "", it will skip the access.
  common::Status GetOrCreateFunctionFromMemory(CUfunction &func,
                                               const std::string &name,
                                               const std::string &ptx_str = "");

private:
  std::unique_ptr<PTXCompilerImpl> impl_;
};

class PTXCompilation {
public:
  static PTXCompilation *GetInstance();

  PTXCompiler *GetCompiler(int device_id);

private:
  PTXCompilation();
  PTXCompiler *GetCompilerImpl(int device_id);
  int dev_count_;
  std::vector<std::unique_ptr<PTXCompiler>> compilers_;
};

} // namespace cuda
} // namespace brt
