//===- conv.h -------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include "brt/core/framework/dtype.h"
#include "brt/core/framework/op_kernel.h"
#include <cudnn.h>

namespace brt {
namespace cuda {

/**
 * Conv Ops
 */
template <typename T> class ConvImpl {
public:
  explicit ConvImpl(const OpAccessor &accessor);

  void Execute(const T *input, const T *filter, T *output, void *workspace,
               cudnnHandle_t handle, cudaStream_t stream);

  size_t GetWorkspaceSize(const ExecutionContext &ctx);

  ~ConvImpl();

private:
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
  cudnnFilterDescriptor_t filter_descriptor;
  cudnnConvolutionDescriptor_t convolution_descriptor;
  bool has_perf_result = false;
  cudnnConvolutionFwdAlgoPerf_t perf;
  const float alpha = 1.f, beta = 0.f;
};

template <typename T>
using Conv = CudnnOpKernelWithWorkspace<ConvImpl<T>, TypedOperand<const T *, 0>,
                                        TypedOperand<const T *, 1>,
                                        TypedOperand<T *, 2>>;

} // namespace cuda
} // namespace brt