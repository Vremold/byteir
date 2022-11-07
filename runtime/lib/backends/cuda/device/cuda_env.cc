//===- cuda_env.cc --------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/backends/cuda/device/cuda_env.h"
#include "brt/backends/cuda/device/common/cuda_call.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace brt {
namespace cuda {
namespace { // TODO: move to cuda common helpers
struct ScopedCudaContext {
  ScopedCudaContext(CUcontext context) {
    BRT_CU_CHECK(cuCtxPushCurrent(context));
  }
  ~ScopedCudaContext() { BRT_CU_CHECK(cuCtxPopCurrent(NULL)); }
};
struct ScopedCudaPrimaryContext {
  ScopedCudaPrimaryContext(int device) : device_(device) {
    BRT_CU_CHECK(cuDevicePrimaryCtxRetain(&ctx_, device));
  }

  CUcontext GetCudaCtx() { return ctx_; }

  ~ScopedCudaPrimaryContext() { cuDevicePrimaryCtxRelease(device_); }

private:
  int device_;
  CUcontext ctx_;
};
} // namespace
CudaEnv::CudaEnv(int device_id) : device_id_(device_id), is_primary_(true) {}

CudaEnv::CudaEnv(CUstream stream) {
  if (stream) {
    CUcontext ctx;
    BRT_CU_CHECK(cuStreamGetCtx(stream, &ctx));
    Initialize(ctx);
  } else { // default stream
    BRT_CUDA_CHECK(cudaGetDevice(&device_id_));
    is_primary_ = true;
  }
}

CudaEnv::CudaEnv(CUcontext context) { Initialize(context); }

void CudaEnv::Initialize(CUcontext context) {
  ctx_ = context;
  ScopedCudaContext guard(context);
  BRT_CU_CHECK(cuCtxGetDevice(&device_id_));
  ScopedCudaPrimaryContext primary_context(device_id_);
  is_primary_ = (primary_context.GetCudaCtx() == context);
}

void CudaEnv::Activate() {
  if (is_primary_) {
    BRT_CUDA_CHECK(cudaSetDevice(device_id_));
  } else {
    CUcontext current;
    BRT_CU_CHECK(cuCtxGetCurrent(&current));
    if (current != ctx_) {
      BRT_CU_CHECK(cuCtxSetCurrent(ctx_));
    }
  }
}
} // namespace cuda
} // namespace brt
