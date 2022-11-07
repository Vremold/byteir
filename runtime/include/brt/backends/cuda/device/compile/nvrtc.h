//===- nvrtc.h ------------------------------------------------*--- C++ -*-===//
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

struct CUDARTCompilationImpl;
class PTXCompilation;

class CUDARTCompilation {
public:
  static CUDARTCompilation *GetInstance();

  ~CUDARTCompilation();

  bool GetLoweredName(const std::string &name, std::string &lowered_name);

  bool GetCachedFunction(CUfunction &func, const std::string &name,
                         int device_id);

  void CreateFunctionFromMemory(CUfunction &func, const std::string &name,
                                int device_id, const std::string &cuda_str);

  void CreateFunctionsFromMemory(std::vector<CUfunction> &funcs,
                                 const std::vector<std::string> &names,
                                 int device_id, const std::string &cuda_str);

  // Get or create a function from a file
  // the file will be accessed if a function is not cached
  // if file = "", it will skip the access.
  common::Status GetOrCreateFunction(CUfunction &func, const std::string &name,
                                     int device_id,
                                     const std::string &file = "");

  // Get or create a function from a file
  // the ptx_str will be accessed if a function is not cached
  // if ptx_str = "", it will skip the access.
  common::Status
  GetOrCreateFunctionFromMemory(CUfunction &func, const std::string &name,
                                int device_id,
                                const std::string &cuda_str = "");

private:
  // private constructor
  CUDARTCompilation();

  PTXCompilation *ptx_handle_;

  CUDARTCompilationImpl *impl_;

  std::vector<char *> nvrtc_opts_;

  bool expect_c_style_;
};

} // namespace cuda
} // namespace brt
