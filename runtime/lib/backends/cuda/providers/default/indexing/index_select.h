//===- index_select.h -----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "./kernels/index_select.h"
#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"

namespace brt {
namespace cuda {

template <typename T> class IndexSelectImpl {
public:
  IndexSelectImpl(const OpAccessor &accessor) {
    auto shape = accessor.GetArgShape(0);
    int ndim = shape.size();
    // parameter dim to specify indexes the input along which demension
    int dim = accessor.GetAttrAsInt("dim");
    A = C = 1;
    for (int i = 0; i < dim; ++i) {
      A *= shape[i];
    }
    input_B = shape[dim];
    output_B = accessor.GetArgShape(1)[0];
    for (int i = dim + 1; i < ndim; ++i) {
      C *= shape[i];
    }
  }

  void Execute(const T *input, const uint32_t *index, T *output,
               cudaStream_t stream) {
    kernel::index_select<T>(input, index, output, A, input_B, output_B, C,
                            stream);
  }

private:
  int A, input_B, output_B, C;
};

template <typename T>
using IndexSelect = CudaOpKernel<IndexSelectImpl<T>,                //
                                 TypedOperand<const T *, 0>,        // input
                                 TypedOperand<const uint32_t *, 1>, // index
                                 TypedOperand<T *, 2>               // output
                                 >;
} // namespace cuda
} // namespace brt