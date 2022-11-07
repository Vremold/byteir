/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v3/includes/utils.h"
#include <cuda_fp16.h>

namespace fastertransformerv3 {
void dense_layer_kernel_launcher(const float *in, const float *weight,
                                 float *out, const int M, const int K,
                                 const int N, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo) {
  const float alpha = 1.0f, beta = 0.0f;
  check_cuda_error(cublasGemmEx(
      cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, weight,
      CUDA_R_32F, N, in, CUDA_R_32F, K, &beta, out, CUDA_R_32F, N, CUDA_R_32F,
      static_cast<cublasGemmAlgo_t>(cublasAlgo)));
}

void dense_layer_kernel_launcher(const __half *in, const __half *weight,
                                 __half *out, const int M, const int K,
                                 const int N, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo) {
  const __half alpha = (__half)1.0f, beta = (__half)0.0f;
  check_cuda_error(cublasGemmEx(
      cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, weight,
      CUDA_R_16F, N, in, CUDA_R_16F, K, &beta, out, CUDA_R_16F, N, CUDA_R_16F,
      static_cast<cublasGemmAlgo_t>(cublasAlgo)));
}

__global__ void add_bias_gelu(float *output, const float *bias, const int M,
                              const int N) {
  int row_offset = blockIdx.x * N;
  for (int tid = threadIdx.x; tid < N; tid += blockDim.x) {
    float out = output[row_offset + tid] + __ldg(&bias[tid]);
    output[row_offset + tid] = gelu(out);
  }
}

__global__ void add_bias_gelu(__half *output, const __half *bias, const int M,
                              const int N) {
  half2 *output_ptr = (half2 *)output;
  const half2 *bias_ptr = (const half2 *)bias;

  int row_offset = blockIdx.x * N / 2;
  for (int tid = threadIdx.x; tid < N / 2; tid += blockDim.x) {
    half2 out = __hadd2(output_ptr[row_offset + tid], __ldg(&bias_ptr[tid]));
    output_ptr[row_offset + tid] = gelu(out);
  }
}

__global__ void add_bias_swish(float *output, const float *bias, const int M,
                               const int N) {
  int row_offset = blockIdx.x * N;
  for (int tid = threadIdx.x; tid < N; tid += blockDim.x) {
    float out = output[row_offset + tid] + __ldg(&bias[tid]);
    output[row_offset + tid] = swish(out);
  }
}

__global__ void add_bias_swish(__half *output, const __half *bias, const int M,
                               const int N) {
  half2 *output_ptr = (half2 *)output;
  const half2 *bias_ptr = (const half2 *)bias;

  int row_offset = blockIdx.x * N / 2;
  for (int tid = threadIdx.x; tid < N / 2; tid += blockDim.x) {
    half2 out = __hadd2(output_ptr[row_offset + tid], __ldg(&bias_ptr[tid]));
    output_ptr[row_offset + tid] = swish(out);
  }
}

__global__ void add_bias_input(float *out, const float *input,
                               const float *bias, int m, int n) {
  int offset = blockIdx.x * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int index = offset + i;
    out[index] = out[index] + __ldg(&input[index]) + __ldg(&bias[i]);
  }
}

__global__ void add_bias_input(__half *out, const __half *input,
                               const __half *bias, int m, int n) {
  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  int id = blockIdx.x * n / 2 + threadIdx.x;
  out_ptr[id] = __hadd2(__hadd2(out_ptr[id], __ldg(&input_ptr[id])),
                        __ldg(&bias_ptr[threadIdx.x]));
}

__global__ void add_bias_input_restore_output(const float *out,
                                              const float *input,
                                              const float *bias, int m, int n,
                                              float *out2, const int *batch_idx,
                                              const int *word_idx) {
  int input_offset = blockIdx.x * n;
  int output_offset = __ldg(&word_idx[blockIdx.x]) * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int index = input_offset + i;
    out2[output_offset + i] =
        out[index] + __ldg(&input[index]) + __ldg(&bias[i]);
  }
}

__global__ void
add_bias_input_restore_output(const __half *out, const __half *input,
                              const __half *bias, int m, int n, __half *out2,
                              const int *batch_idx, const int *word_idx) {
  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  int offset = __ldg(&word_idx[blockIdx.x]);
  int id = blockIdx.x * n / 2 + threadIdx.x;
  ((half2 *)out2)[offset * n / 2 + threadIdx.x] =
      __hadd2(__hadd2(out_ptr[id], __ldg(&input_ptr[id])),
              __ldg(&bias_ptr[threadIdx.x]));
}

__global__ void add_bias_half_input(float *out, const float *input,
                                    const float *bias, int m, int n) {
  int offset = blockIdx.x * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int index = offset + i;
    out[index] = __ldg(&input[index]) + (out[index] + __ldg(&bias[i])) * 0.5f;
  }
}

__global__ void add_bias_half_input(__half *out, const __half *input,
                                    const __half *bias, int m, int n) {
  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  int id = blockIdx.x * n / 2 + threadIdx.x;
  out_ptr[id] =
      __hadd2(__ldg(&input_ptr[id]),
              __hmul2(__hadd2(out_ptr[id], __ldg(&bias_ptr[threadIdx.x])),
                      half2(0.5f, 0.5f)));
}

__global__ void
add_bias_half_input_restore_output(const float *out, const float *input,
                                   const float *bias, int m, int n, float *out2,
                                   const int *batch_idx, const int *word_idx) {
  int input_offset = blockIdx.x * n;
  int output_offset = __ldg(&word_idx[blockIdx.x]) * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int index = input_offset + i;
    out2[output_offset + i] =
        __ldg(&input[index]) + (__ldg(&out[index]) + __ldg(&bias[i])) * 0.5f;
  }
}

__global__ void add_bias_half_input_restore_output(
    const __half *out, const __half *input, const __half *bias, int m, int n,
    __half *out2, const int *batch_idx, const int *word_idx) {
  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  int offset = __ldg(&word_idx[blockIdx.x]);
  int id = blockIdx.x * n / 2 + threadIdx.x;
  ((half2 *)out2)[offset * n / 2 + threadIdx.x] = __hadd2(
      __ldg(&input_ptr[id]),
      __hmul2(__hadd2(__ldg(&out_ptr[id]), __ldg(&bias_ptr[threadIdx.x])),
              half2(0.5f, 0.5f)));
}

} // namespace fastertransformerv3
