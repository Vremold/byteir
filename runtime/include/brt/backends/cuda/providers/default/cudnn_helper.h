//===- cudnn_helper.h -----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/common/common.h"
#include "brt/core/framework/dtype.h"

namespace brt {

inline cudnnDataType_t ConvertBRTDTypeToCudnnDtype(DTypeEnum dataType) {
  if (dataType == DTypeEnum::Float32) {
    return CUDNN_DATA_FLOAT;
  } else if (dataType == DTypeEnum::Float16) {
    return CUDNN_DATA_HALF;
  } else {
    BRT_THROW("invalid data type");
  }
}

inline const char *cudnn_math_type_to_str(cudnnMathType_t mathType) {
  switch (mathType) {
  case CUDNN_DEFAULT_MATH:
    return "CUDNN_DEFAULT_MATH";
  case CUDNN_TENSOR_OP_MATH:
    return "CUDNN_TENSOR_OP_MATH";
  case CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION:
    return "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION";
  case CUDNN_FMA_MATH:
    return "CUDNN_FMA_MATH";
  default:
    break;
  }
  return "Unknown Math Type";
}

inline const char *cudnn_fwd_algo_to_str(cudnnConvolutionFwdAlgo_t algo) {
  switch (algo) {
  case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
    return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
  case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
    return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM";
  case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
    return "CUDNN_CONVOLUTION_FWD_ALGO_GEMM";
  case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
    return "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT";
  case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
    return "CUDNN_CONVOLUTION_FWD_ALGO_FFT";
  case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
    return "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING";
  case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
    return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD";
  case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
    return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED";
  default:
    break;
  }
  return "Unknown Conv Fwd Algo";
}

inline const char *
cudnn_bwd_data_algo_to_str(cudnnConvolutionBwdDataAlgo_t algo) {
  switch (algo) {
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
    return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
    return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
    return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
    return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
    return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
    return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED";
  default:
    break;
  }
  return "Unknown Conv Bwd Data Algo";
}

} // namespace brt
