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

__global__ void add_bias_half_input_layernorm(float *out, const float *input,
                                              const float *bias,
                                              const void *gamma,
                                              const void *beta, int m, int n,
                                              bool use_fp32) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;

  float local_out = (out[bid * n + tid] + __ldg(&bias[tid])) * 0.5f +
                    __ldg(&input[bid * n + tid]);

  float mean = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  local_out -= s_mean;
  float variance = blockReduceSum<float>(local_out * local_out);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  out[bid * n + tid] = local_out * s_variance * __ldg(&((float *)gamma)[tid]) +
                       __ldg(&((float *)beta)[tid]);
}

__global__ void add_bias_half_input_layernorm(__half *out, const __half *input,
                                              const __half *bias,
                                              const void *gamma,
                                              const void *beta, int m, int n,
                                              bool use_fp32) {
  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;

  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  int id = blockIdx.x * n / 2 + tid;
  float2 local_out_fp2 = __half22float2(__hadd2(
      __hmul2(__hadd2(out_ptr[id], __ldg(&bias_ptr[tid])), half2(0.5f, 0.5f)),
      __ldg(&input_ptr[id])));
  float local_out = local_out_fp2.x + local_out_fp2.y;

  float mean = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  local_out_fp2.x -= s_mean;
  local_out_fp2.y -= s_mean;
  float variance =
      local_out_fp2.x * local_out_fp2.x + local_out_fp2.y * local_out_fp2.y;
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  float2 gamma_val, beta_val;
  if (use_fp32) {
    gamma_val = __ldg(&((const float2 *)gamma)[tid]);
    beta_val = __ldg(&((const float2 *)beta)[tid]);
  } else {
    gamma_val = __half22float2(__ldg(&((const half2 *)gamma)[tid]));
    beta_val = __half22float2(__ldg(&((const half2 *)beta)[tid]));
  }

  local_out_fp2.x = local_out_fp2.x * s_variance * gamma_val.x + beta_val.x;
  local_out_fp2.y = local_out_fp2.y * s_variance * gamma_val.y + beta_val.y;
  out_ptr[id] = __float22half2_rn(local_out_fp2);
}

__global__ void add_bias_half_input_layernorm_restore_output(
    const float *out, const float *input, const float *bias, const void *gamma,
    const void *beta, int m, int n, bool use_fp32, float *out2,
    const int *batch_idx, const int *word_idx) {
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;

  float local_out = (out[blockIdx.x * n + tid] + __ldg(&bias[tid])) * 0.5f +
                    __ldg(&input[blockIdx.x * n + tid]);

  float mean = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  local_out -= s_mean;
  float variance = blockReduceSum<float>(local_out * local_out);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  int offset = __ldg(&word_idx[blockIdx.x]);
  out2[offset * n + tid] =
      local_out * s_variance * __ldg(&((float *)gamma)[tid]) +
      __ldg(&((float *)beta)[tid]);
}

__global__ void add_bias_half_input_layernorm_restore_output(
    const __half *out, const __half *input, const __half *bias,
    const void *gamma, const void *beta, int m, int n, bool use_fp32,
    __half *out2, const int *batch_idx, const int *word_idx) {
  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;

  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  int id = blockIdx.x * n / 2 + tid;
  float2 local_out_fp2 = __half22float2(__hadd2(
      __hmul2(__hadd2(out_ptr[id], __ldg(&bias_ptr[tid])), half2(0.5f, 0.5f)),
      __ldg(&input_ptr[id])));
  float local_out = local_out_fp2.x + local_out_fp2.y;

  float mean = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  local_out_fp2.x -= s_mean;
  local_out_fp2.y -= s_mean;
  float variance =
      local_out_fp2.x * local_out_fp2.x + local_out_fp2.y * local_out_fp2.y;
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  float2 gamma_val, beta_val;
  if (use_fp32) {
    gamma_val = __ldg(&((const float2 *)gamma)[tid]);
    beta_val = __ldg(&((const float2 *)beta)[tid]);
  } else {
    gamma_val = __half22float2(__ldg(&((const half2 *)gamma)[tid]));
    beta_val = __half22float2(__ldg(&((const half2 *)beta)[tid]));
  }

  local_out_fp2.x = local_out_fp2.x * s_variance * gamma_val.x + beta_val.x;
  local_out_fp2.y = local_out_fp2.y * s_variance * gamma_val.y + beta_val.y;

  int offset = __ldg(&word_idx[blockIdx.x]);
  ((half2 *)out2)[offset * n / 2 + tid] = __float22half2_rn(local_out_fp2);
}

template <const int ite>
__global__ void add_bias_half_input_layernorm_v2(float *out, const float *input,
                                                 const float *bias,
                                                 const void *gamma,
                                                 const void *beta, int m, int n,
                                                 bool use_fp32) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = bid * n + col_id;
    local_out[i] = (out[id] + __ldg(&bias[col_id])) * 0.5f + __ldg(&input[id]);
    sum += local_out[i];
  }

  float mean = blockReduceSum<float>(sum);
  if (threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float var = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out[i] -= s_mean;
    var += local_out[i] * local_out[i];
  }

  float variance = blockReduceSum<float>(var);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = bid * n + col_id;
    out[id] = local_out[i] * s_variance * __ldg(&((float *)gamma)[col_id]) +
              __ldg(&((float *)beta)[col_id]);
  }
}

template <const int ite>
__global__ void add_bias_half_input_layernorm_v2(
    __half *out, const __half *input, const __half *bias, const void *gamma,
    const void *beta, int m, int n, bool use_fp32) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;

  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  float sum = 0.0f;
  float2 local_out_fp2[ite];
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = bid * n / 2 + col_id;
    local_out_fp2[i] = __half22float2(
        __hadd2(__hmul2(__hadd2(out_ptr[id], __ldg(&bias_ptr[col_id])),
                        half2(0.5f, 0.5f)),
                __ldg(&input_ptr[id])));
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  sum = blockReduceSum<float>(sum);
  if (threadIdx.x == 0)
    s_mean = sum / n;
  __syncthreads();

  float variance = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out_fp2[i].x -= s_mean;
    local_out_fp2[i].y -= s_mean;
    variance += local_out_fp2[i].x * local_out_fp2[i].x +
                local_out_fp2[i].y * local_out_fp2[i].y;
  }

  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  float2 gamma_val[ite], beta_val[ite];
  if (use_fp32) {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * blockDim.x + tid;
      gamma_val[i] = __ldg(&((const float2 *)gamma)[col_id]);
      beta_val[i] = __ldg(&((const float2 *)beta)[col_id]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * blockDim.x + tid;
      gamma_val[i] = __half22float2(__ldg(&((const half2 *)gamma)[col_id]));
      beta_val[i] = __half22float2(__ldg(&((const half2 *)beta)[col_id]));
    }
  }

#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = bid * n / 2 + col_id;
    local_out_fp2[i].x =
        local_out_fp2[i].x * s_variance * gamma_val[i].x + beta_val[i].x;
    local_out_fp2[i].y =
        local_out_fp2[i].y * s_variance * gamma_val[i].y + beta_val[i].y;
    out_ptr[id] = __float22half2_rn(local_out_fp2[i]);
  }
}

template <const int ite>
__global__ void add_bias_half_input_layernorm_restore_output_v2(
    const float *out, const float *input, const float *bias, const void *gamma,
    const void *beta, int m, int n, bool use_fp32, float *out2,
    const int *batch_idx, const int *word_idx) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = bid * n + col_id;
    local_out[i] = (out[id] + __ldg(&bias[col_id])) * 0.5f + __ldg(&input[id]);
    sum += local_out[i];
  }

  float mean = blockReduceSum<float>(sum);
  if (threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float var = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out[i] -= s_mean;
    var += local_out[i] * local_out[i];
  }

  float variance = blockReduceSum<float>(var);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  int offset = __ldg(&word_idx[bid]);
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = offset * n + col_id;
    out2[id] = local_out[i] * s_variance * __ldg(&((float *)gamma)[col_id]) +
               __ldg(&((float *)beta)[col_id]);
  }
}

template <const int ite>
__global__ void add_bias_half_input_layernorm_restore_output_v2(
    const __half *out, const __half *input, const __half *bias,
    const void *gamma, const void *beta, int m, int n, bool use_fp32,
    __half *out2, const int *batch_idx, const int *word_idx) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;

  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  float sum = 0.0f;
  float2 local_out_fp2[ite];
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = bid * n / 2 + col_id;
    local_out_fp2[i] = __half22float2(
        __hadd2(__hmul2(__hadd2(out_ptr[id], __ldg(&bias_ptr[col_id])),
                        half2(0.5f, 0.5f)),
                __ldg(&input_ptr[id])));
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  sum = blockReduceSum<float>(sum);
  if (threadIdx.x == 0)
    s_mean = sum / n;
  __syncthreads();

  float variance = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out_fp2[i].x -= s_mean;
    local_out_fp2[i].y -= s_mean;
    variance += local_out_fp2[i].x * local_out_fp2[i].x +
                local_out_fp2[i].y * local_out_fp2[i].y;
  }

  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  float2 gamma_val[ite], beta_val[ite];
  if (use_fp32) {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * blockDim.x + tid;
      gamma_val[i] = __ldg(&((const float2 *)gamma)[col_id]);
      beta_val[i] = __ldg(&((const float2 *)beta)[col_id]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * blockDim.x + tid;
      gamma_val[i] = __half22float2(__ldg(&((const half2 *)gamma)[col_id]));
      beta_val[i] = __half22float2(__ldg(&((const half2 *)beta)[col_id]));
    }
  }

  int offset = __ldg(&word_idx[blockIdx.x]);
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = offset * n / 2 + col_id;
    local_out_fp2[i].x =
        local_out_fp2[i].x * s_variance * gamma_val[i].x + beta_val[i].x;
    local_out_fp2[i].y =
        local_out_fp2[i].y * s_variance * gamma_val[i].y + beta_val[i].y;
    ((half2 *)out2)[id] = __float22half2_rn(local_out_fp2[i]);
  }
}

__global__ void add_bias_half_input_out_layernorm(
    float *out, const float *input, const float *bias, float *out2,
    const void *gamma, const void *beta, int m, int n, bool use_fp32) {
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;

  float local_out = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    int offset = blockIdx.x * n + i;
    float temp = (out[offset] + __ldg(&bias[i])) * 0.5f + __ldg(&input[offset]);
    out2[offset] = temp;
    local_out += temp;
  }

  float mean = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  local_out -= s_mean;
  float variance = blockReduceSum<float>(local_out * local_out);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x)
    out[blockIdx.x * n + i] =
        local_out * s_variance * __ldg(&((float *)gamma)[i]) +
        __ldg(&((float *)beta)[i]);
}

__global__ void add_bias_half_input_out_layernorm(
    __half *out, const __half *input, const __half *bias, __half *out2,
    const void *gamma, const void *beta, int m, int n, bool use_fp32) {
  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;

  half2 *out_ptr = (half2 *)out;
  half2 *out2_ptr = (half2 *)out2;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  int id = blockIdx.x * n / 2 + tid;
  half2 temp = __hadd2(
      __hmul2(__hadd2(out_ptr[id], __ldg(&bias_ptr[tid])), half2(0.5f, 0.5f)),
      __ldg(&input_ptr[id]));
  out2_ptr[id] = temp;
  float2 local_out_fp2 = __half22float2(temp);
  float local_out = local_out_fp2.x + local_out_fp2.y;

  float mean = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  local_out_fp2.x -= s_mean;
  local_out_fp2.y -= s_mean;
  float variance =
      local_out_fp2.x * local_out_fp2.x + local_out_fp2.y * local_out_fp2.y;
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  float2 gamma_val, beta_val;
  if (use_fp32) {
    gamma_val = __ldg(&((const float2 *)gamma)[tid]);
    beta_val = __ldg(&((const float2 *)beta)[tid]);
  } else {
    gamma_val = __half22float2(__ldg(&((const half2 *)gamma)[tid]));
    beta_val = __half22float2(__ldg(&((const half2 *)beta)[tid]));
  }

  local_out_fp2.x = local_out_fp2.x * s_variance * gamma_val.x + beta_val.x;
  local_out_fp2.y = local_out_fp2.y * s_variance * gamma_val.y + beta_val.y;
  out_ptr[id] = __float22half2_rn(local_out_fp2);
}

template <const int ite>
__global__ void add_bias_half_input_out_layernorm_v2(
    float *out, const float *input, const float *bias, float *out2,
    const void *gamma, const void *beta, int m, int n, bool use_fp32) {
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = blockIdx.x * n + col_id;
    local_out[i] = (out[id] + __ldg(&bias[col_id])) * 0.5f + __ldg(&input[id]);
    out2[id] = local_out[i];
    sum += local_out[i];
  }

  float mean = blockReduceSum<float>(sum);
  if (threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float var = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out[i] -= s_mean;
    var += local_out[i] * local_out[i];
  }

  float variance = blockReduceSum<float>(var);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = blockIdx.x * n + col_id;
    out[id] = local_out[i] * s_variance * __ldg(&((float *)gamma)[col_id]) +
              __ldg(&((float *)beta)[col_id]);
  }
}

template <const int ite>
__global__ void add_bias_half_input_out_layernorm_v2(
    __half *out, const __half *input, const __half *bias, __half *out2,
    const void *gamma, const void *beta, int m, int n, bool use_fp32) {
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;

  half2 *out_ptr = (half2 *)out;
  half2 *out2_ptr = (half2 *)out2;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  float sum = 0.0f;
  float2 local_out_fp2[ite];
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = blockIdx.x * n / 2 + col_id;
    half2 temp = __hadd2(__hmul2(__hadd2(out_ptr[id], __ldg(&bias_ptr[col_id])),
                                 half2(0.5f, 0.5f)),
                         __ldg(&input_ptr[id]));
    out2_ptr[id] = temp;
    local_out_fp2[i] = __half22float2(temp);
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  sum = blockReduceSum<float>(sum);
  if (threadIdx.x == 0)
    s_mean = sum / n;
  __syncthreads();

  float variance = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out_fp2[i].x -= s_mean;
    local_out_fp2[i].y -= s_mean;
    variance += local_out_fp2[i].x * local_out_fp2[i].x +
                local_out_fp2[i].y * local_out_fp2[i].y;
  }

  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  float2 gamma_val[ite], beta_val[ite];
  if (use_fp32) {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * blockDim.x + tid;
      gamma_val[i] = __ldg(&((const float2 *)gamma)[col_id]);
      beta_val[i] = __ldg(&((const float2 *)beta)[col_id]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * blockDim.x + tid;
      gamma_val[i] = __half22float2(__ldg(&((const half2 *)gamma)[col_id]));
      beta_val[i] = __half22float2(__ldg(&((const half2 *)beta)[col_id]));
    }
  }

#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = blockIdx.x * n / 2 + col_id;
    local_out_fp2[i].x =
        local_out_fp2[i].x * s_variance * gamma_val[i].x + beta_val[i].x;
    local_out_fp2[i].y =
        local_out_fp2[i].y * s_variance * gamma_val[i].y + beta_val[i].y;
    out_ptr[id] = __float22half2_rn(local_out_fp2[i]);
  }
}

template <typename T>
void add_bias_half_input_layernorm_kernel_launcher(
    T *output, const T *input, const T *bias, const void *gamma,
    const void *beta, int m, int n, int hidden_dim, cudaStream_t stream,
    bool use_fp32) {
  dim3 grid(m), block(hidden_dim);

  if (m >= 256 && (n == 512 || n == 768 || n == 1024)) {
    const int ite = 4;
    add_bias_half_input_layernorm_v2<ite><<<grid, block.x / ite, 0, stream>>>(
        output, input, bias, gamma, beta, m, n, use_fp32);
  } else
    add_bias_half_input_layernorm<<<grid, block, 0, stream>>>(
        output, input, bias, gamma, beta, m, n, use_fp32);
}

template <typename T>
void add_bias_half_input_layernorm_restore_output_kernel_launcher(
    T *output, const T *input, const T *bias, const void *gamma,
    const void *beta, int m, int n, int hidden_dim, cudaStream_t stream,
    bool use_fp32, T *output2, int *batch_idx, int *word_idx) {
  dim3 grid(m), block(hidden_dim);

  if (m >= 256 && (n == 512 || n == 768 || n == 1024)) {
    const int ite = 4;
    add_bias_half_input_layernorm_restore_output_v2<ite>
        <<<grid, block.x / ite, 0, stream>>>(output, input, bias, gamma, beta,
                                             m, n, use_fp32, output2, batch_idx,
                                             word_idx);
  } else
    add_bias_half_input_layernorm_restore_output<<<grid, block, 0, stream>>>(
        output, input, bias, gamma, beta, m, n, use_fp32, output2, batch_idx,
        word_idx);
}

template <typename T>
void add_bias_half_input_out_layernorm_kernel_launcher(
    T *output, const T *input, const T *bias, T *output2, const void *gamma,
    const void *beta, int m, int n, int hidden_dim, cudaStream_t stream,
    bool use_fp32) {
  dim3 grid(m), block(hidden_dim);

  if (m >= 256 && (n == 512 || n == 768 || n == 1024)) {
    const int ite = 4;
    add_bias_half_input_out_layernorm_v2<ite>
        <<<grid, block.x / ite, 0, stream>>>(output, input, bias, output2,
                                             gamma, beta, m, n, use_fp32);
  } else
    add_bias_half_input_out_layernorm<<<grid, block, 0, stream>>>(
        output, input, bias, output2, gamma, beta, m, n, use_fp32);
}
} // namespace fastertransformerv3