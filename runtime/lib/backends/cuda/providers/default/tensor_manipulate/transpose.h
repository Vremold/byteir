//===- transpose.h --------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include <cudnn.h>

namespace brt {
namespace cuda {

/**
 * TransposeBase
 */
template <typename T> class TransposeBase {
public:
  virtual void Execute(const T *input, T *output, cudnnHandle_t handle,
                       cudaStream_t stream) = 0;
  virtual ~TransposeBase() = default;
};

/**
 * Transpose2D
 */
template <typename T> class Transpose2D : public TransposeBase<T> {
public:
  explicit Transpose2D(const OpAccessor &accessor);

  virtual void Execute(const T *input, T *output, cudnnHandle_t handle,
                       cudaStream_t stream) override;

private:
  std::vector<int64_t> input_shape;
};

/**
 * Transpose4D
 */
template <typename T> class Transpose4D : public TransposeBase<T> {
public:
  explicit Transpose4D(const OpAccessor &accessor);

  virtual void Execute(const T *input, T *output, cudnnHandle_t handle,
                       cudaStream_t stream) override;

  virtual ~Transpose4D();

private:
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
};

/**
 * TransposeImpl
 */
template <typename T> class TransposeImpl {
public:
  explicit TransposeImpl(const OpAccessor &accessor);

  void Execute(const T *input, T *output, cudnnHandle_t handle,
               cudaStream_t stream);

  ~TransposeImpl();

private:
  TransposeBase<T> *impl = nullptr;
};

template <typename T>
using Transpose = CudnnOpKernel<TransposeImpl<T>, TypedOperand<const T *, 0>,
                                TypedOperand<T *, 1>>;
} // namespace cuda
} // namespace brt
