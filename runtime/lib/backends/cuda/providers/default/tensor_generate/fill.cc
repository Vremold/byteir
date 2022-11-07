//===- fill.cc ------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "./fill.h"
#include "./kernels/fill.h"

#include "brt/backends/cuda/device/common/util.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/core/common/common.h"
#include "brt/core/framework/op_accessor.h"

#include <cuda_fp16.h>

namespace brt {
namespace cuda {
FillOpKernel::FillOpKernel(const OpKernelInfo &info)
    : OpKernel(info, false, false, true, true) {}

// FIXME: implement as callonce
common::Status FillOpKernel::RunImpl(const ExecutionContext &ctx) {
  // TODO: explicit synchronization between h2d(frame initialize/session
  // initialize) stream and compute stream?
  OpAccessor accessor(info_, ctx.exec_frame);
  DTypeEnum dtype = accessor.GetArgDTypeEnum(0);
  cudaStream_t stream =
      static_cast<CUDAWorkQueue *>(ctx.work_queue)->GetComputeStream();
  void *device_p = accessor.GetArgAsyncValueRef(0);
  size_t length = accessor.GetNumElementsOfShape(accessor.GetArgShape(0));
  // TODO: common helper for dtype dispatch
#define CASE(dtype, ctype, mlir_type)                                          \
  case DTypeEnum::dtype:                                                       \
    kernel::Fill(                                                              \
        stream, static_cast<ctype *>(device_p),                                \
        static_cast<ctype>(accessor.GetAttrAsSplatValue<mlir_type>("value")),  \
        length);                                                               \
    return common::Status::OK()
  switch (dtype) {
    CASE(Float32, float, float);
    CASE(Int64, int64_t, int64_t);
    CASE(Float64, double, double);
    CASE(Float16, __half, float);
#undef CASE
  default:
    return common::Status(common::StatusCategory::BRT,
                          common::StatusCode::NOT_IMPLEMENTED,
                          "not supported dtype");
  };
  return common::Status::OK();
}

common::Status FillOpKernel::ProloguePerFrame(const ExecutionContext &) {
  return common::Status::OK();
}

common::Status FillOpKernel::EpiloguePerFrame(const ExecutionContext &) {
  return common::Status::OK();
}

} // namespace cuda
} // namespace brt
