/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "fastertransformer_v3/includes/common.h"
#include <cuda_fp16.h>

namespace fastertransformerv3 {
__global__ void add_bias_glu(const float *input, const float *bias,
                             float *output, const int M, const int N);

__global__ void add_bias_glu(const __half *input, const __half *bias,
                             __half *output, const int M, const int N);

__global__ void transpose_to_NCL(const float *input, float *output,
                                 const int seq_len, const int hidden_dim);

__global__ void transpose_to_NCL(const __half *input, __half *output,
                                 const int seq_len, const int hidden_dim);

__global__ void transpose_to_NLC(const float *input, float *output,
                                 const int hidden_dim, const int seq_len);

__global__ void transpose_to_NLC(const __half *input, __half *output,
                                 const int hidden_dim, const int seq_len);

// __global__
// void add_bias_glu_transpose_dim12(const float* input, const float* bias,
// float* output, const int M, const int N);

// __global__
// void add_bias_glu_transpose_dim12(const __half* input, const __half* bias,
// __half* output, const int M, const int N);

__global__ void depthwise_conv(const float *input, const float *conv_kernel,
                               float *output, const int seq_len, const int N);

__global__ void depthwise_conv(const __half *input, const __half *conv_kernel,
                               __half *output, const int seq_len, const int N);

__global__ void add_bias_batchnorm_swish(float *output, const float *bias,
                                         const float *mean, const float *var,
                                         const void *gamma, const void *beta,
                                         int m, int n, int hidden_dim,
                                         bool use_fp32);

__global__ void add_bias_batchnorm_swish(__half *output, const __half *bias,
                                         const __half *mean, const __half *var,
                                         const void *gamma, const void *beta,
                                         int m, int n, int hidden_dim,
                                         bool use_fp32);

// __global__
// void transpose_dim12_add_bias_batchnorm_swish(float* output, const float*
// bias, const float* mean, const float* var, const void* gamma, const void*
// beta, int m, int n, int hidden_dim, bool use_fp32);

// __global__
// void transpose_dim12_add_bias_batchnorm_swish(__half* output, const __half*
// bias, const __half* mean, const __half* var, const void* gamma, const void*
// beta, int m, int n, int hidden_dim, bool use_fp32);
} // namespace fastertransformerv3