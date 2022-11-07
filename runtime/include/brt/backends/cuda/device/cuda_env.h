//===- cuda_env.h ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

struct CUctx_st;
struct CUstream_st;

namespace brt {
namespace cuda {
class CudaEnv {
public:
  CudaEnv(int device_id);
  CudaEnv(CUctx_st *ctx);
  CudaEnv(CUstream_st *stream);

  void Activate();

  // return true if this CudaEnv is associated with the cuda primary context
  bool IsPrimaryContext() { return is_primary_; }

  int GetDeviceID() { return device_id_; }

private:
  void Initialize(CUctx_st *st);

  int device_id_;
  bool is_primary_;
  CUctx_st *ctx_;
};
} // namespace cuda
} // namespace brt
