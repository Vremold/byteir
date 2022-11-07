//===- index_put.h --------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include "kernels/index_put.h"

namespace brt {
namespace cuda {

template <typename T> class IndexPutImpl {
public:
  IndexPutImpl(const OpAccessor &accessor) {
    auto shape = accessor.GetArgShape(0);
    int ndim = shape.size();
    // parameter dim to specify indexes the input along which demension
    int dim = accessor.GetAttrAsInt("dim");
    total_size = 1;
    for (int i = 0; i <= dim; ++i) {
      total_size *= shape[i];
    }
    index_bound = accessor.GetArgShape(1)[0];
    feature_bound = 1;
    for (int i = dim + 1; i < ndim; ++i) {
      feature_bound *= shape[i];
    }
    total_size *= feature_bound;
  }

  void Execute(const T *input, const int64_t *index, const T *update, T *output,
               cudaStream_t stream) {
    kernel::index_put<T, true>(input, index, update, output, index_bound,
                               feature_bound, total_size, stream);
  }

private:
  int index_bound;
  int feature_bound;
  int total_size;
};

template <typename T>
using IndexPut = CudaOpKernel<IndexPutImpl<T>,                  //
                              TypedOperand<const T *, 0>,       // input
                              TypedOperand<const int64_t *, 1>, // index
                              TypedOperand<const T *, 2>,       // update
                              TypedOperand<T *, 3>              // output
                              >;
} // namespace cuda
} // namespace brt