//===- rng.h --------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "./kernels/rng.h"
#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include "brt/core/framework/op_accessor.h"

namespace brt {
namespace cuda {
class RngImplBase {
protected:
  RngImplBase(const OpAccessor &accessor) {
    auto shape = accessor.GetArgShape(0);
    nr_elems = accessor.GetNumElementsOfShape(shape);

    auto dtype = accessor.GetArgDTypeEnum(0);
    BRT_ENFORCE(dtype == DTypeEnum::Float32, "only float32 is supported now");
  }

  size_t nr_elems;
};

class RngUniformImpl : public RngImplBase {
public:
  RngUniformImpl(const OpAccessor &accessor) : RngImplBase(accessor) {
    if (accessor.HasAttr("low")) {
      BRT_ENFORCE(accessor.HasAttr("high"));
      low = accessor.GetAttrAsFloat("low");
      high = accessor.GetAttrAsFloat("high");
      BRT_ENFORCE(low < high, "invalid uniform rng attributes");
    } else {
      BRT_ENFORCE(!accessor.HasAttr("high"));
      low = 0;
      high = 1;
    }
  }

  void Execute(float *ptr, curandGenerator_t generator, cudaStream_t stream) {
    if (low == 0 && high == 1) {
      BRT_CURAND_CHECK(curandGenerateUniform(generator, ptr, nr_elems));
    } else {
      kernel::RngUniform(stream, ptr, nr_elems, low, high);
    }
  }

private:
  float low, high;
};

class RngNormalImpl : public RngImplBase {
public:
  RngNormalImpl(const OpAccessor &accessor) : RngImplBase(accessor) {
    mean = accessor.GetAttrAsFloat("mean");
    stddev = accessor.GetAttrAsFloat("stddev");
  }

  void Execute(float *ptr, curandGenerator_t generator, cudaStream_t) {
    BRT_CURAND_CHECK(
        curandGenerateNormal(generator, ptr, nr_elems, mean, stddev));
  }

private:
  float mean, stddev;
};

using RngUniform = CurandOpKernel<RngUniformImpl, TypedOperand<float *, 0>>;
using RngNormal = CurandOpKernel<RngNormalImpl, TypedOperand<float *, 0>>;

} // namespace cuda
} // namespace brt
