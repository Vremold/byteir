/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "common.h"

namespace fastertransformerv4 {
void dense_layer_kernel_launcher(const float *A, const float *B, float *out,
                                 const int M, const int K, const int N,
                                 cublasOperation_t trans_A,
                                 cublasOperation_t trans_B, float alpha,
                                 float beta, cublasHandle_t cublas_handle,
                                 int cublasAlgo = -1);

void dense_layer_kernel_launcher(const __half *A, const __half *B, __half *out,
                                 const int M, const int K, const int N,
                                 cublasOperation_t trans_A,
                                 cublasOperation_t trans_B, __half alpha,
                                 __half beta, cublasHandle_t cublas_handle,
                                 int cublasAlgo = 99);

void cublas_Gemm_Strided_Batched(const float *A, const float *B, float *out,
                                 const int M, const int K, const int N,
                                 const int batch_count,
                                 cublasOperation_t trans_A,
                                 cublasOperation_t trans_B, float alpha,
                                 float beta, cublasHandle_t cublas_handle,
                                 int cublasAlgo = -1);

void cublas_Gemm_Strided_Batched(const __half *A, const __half *B, __half *out,
                                 const int M, const int K, const int N,
                                 const int batch_count,
                                 cublasOperation_t trans_A,
                                 cublasOperation_t trans_B, __half alpha,
                                 __half beta, cublasHandle_t cublas_handle,
                                 int cublasAlgo = 99);
} // namespace fastertransformerv4