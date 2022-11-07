/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v3/includes/layernorm_kernels.h"
#include "fastertransformer_v3/includes/utils.h"
#include <cuda_fp16.h>

namespace fastertransformerv3
{

__global__
void input_layernorm(
                float *out, const float *input,
                const void *gamma, const void *beta, int m, int n, bool use_fp32)
{
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;

    float local_out = __ldg(&input[blockIdx.x * n + tid]);

    float mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    local_out -= s_mean;
    float variance = blockReduceSum<float>(local_out * local_out);
    if(threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6f);
    __syncthreads();

    for(int i = tid; i < n; i += blockDim.x)
        out[blockIdx.x * n + i] = local_out * s_variance * __ldg(&((float *)gamma)[i]) + __ldg(&((float *)beta)[i]);
}

__global__
void input_layernorm(
                __half *out, const __half *input,
                const void *gamma, const void *beta, int m, int n, bool use_fp32)
{
    int tid = threadIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;

    half2 *out_ptr = (half2 *)out;
    const half2 *input_ptr = (const half2 *)input;

    int id = blockIdx.x * n / 2 + tid;
    float2 local_out_fp2 = __half22float2(__ldg(&input_ptr[id]));

    float local_out = local_out_fp2.x + local_out_fp2.y;

    float mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    local_out_fp2.x -= s_mean;
    local_out_fp2.y -= s_mean;
    float variance = local_out_fp2.x * local_out_fp2.x + local_out_fp2.y * local_out_fp2.y;
    variance = blockReduceSum<float>(variance);
    if(threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6f);
    __syncthreads();

    float2 gamma_val, beta_val;
    if(use_fp32)
    {
        gamma_val = __ldg(&((const float2 *)gamma)[tid]);
        beta_val  = __ldg(&((const float2 *)beta)[tid]);
    }
    else
    {
        gamma_val = __half22float2(__ldg(&((const half2 *)gamma)[tid]));
        beta_val  = __half22float2(__ldg(&((const half2 *)beta)[tid]));
    }

    local_out_fp2.x = local_out_fp2.x * s_variance * gamma_val.x + beta_val.x;
    local_out_fp2.y = local_out_fp2.y * s_variance * gamma_val.y + beta_val.y;
    out_ptr[id] = __float22half2_rn(local_out_fp2);
}

__global__
void input_compress_layernorm(
                float *out, const float *input,
                const void *gamma, const void *beta, int m, int n, bool use_fp32,
                float *out2, const int *batch_idx, const int *word_idx)
{
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;

    int offset = __ldg(&word_idx[blockIdx.x]);

    float local_out = __ldg(&input[offset * n + tid]);
    out[blockIdx.x * n + tid] = local_out;

    float mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    local_out -= s_mean;
    float variance = blockReduceSum<float>(local_out * local_out);
    if(threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6f);
    __syncthreads();

    out2[blockIdx.x * n + tid] = local_out * s_variance * __ldg(&((float *)gamma)[tid]) + __ldg(&((float *)beta)[tid]);
}

__global__
void input_compress_layernorm(
                __half *out, const __half *input,
                const void *gamma, const void *beta, int m, int n, bool use_fp32,
                __half *out2, const int *batch_idx, const int *word_idx)
{
    int tid = threadIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;

    const half2 *input_ptr = (const half2 *)input;

    int offset = __ldg(&word_idx[blockIdx.x]);

    int id = offset * n / 2 + tid;
    half2 temp = __ldg(&input_ptr[id]);
    ((half2 *)out)[blockIdx.x * n / 2 + tid] = temp;
    float2 local_out_fp2 = __half22float2(temp);

    float local_out = local_out_fp2.x + local_out_fp2.y;

    float mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    local_out_fp2.x -= s_mean;
    local_out_fp2.y -= s_mean;
    float variance = local_out_fp2.x * local_out_fp2.x + local_out_fp2.y * local_out_fp2.y;
    variance = blockReduceSum<float>(variance);
    if(threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6f);
    __syncthreads();

    float2 gamma_val, beta_val;
    if(use_fp32)
    {
        gamma_val = __ldg(&((const float2 *)gamma)[tid]);
        beta_val  = __ldg(&((const float2 *)beta)[tid]);
    }
    else
    {
        gamma_val = __half22float2(__ldg(&((const half2 *)gamma)[tid]));
        beta_val  = __half22float2(__ldg(&((const half2 *)beta)[tid]));
    }

    local_out_fp2.x = local_out_fp2.x * s_variance * gamma_val.x + beta_val.x;
    local_out_fp2.y = local_out_fp2.y * s_variance * gamma_val.y + beta_val.y;

    ((half2 *)out2)[blockIdx.x * n / 2 + tid] = __float22half2_rn(local_out_fp2);
}

__global__
void add_bias_input_layernorm(
                float *out, const float *input, const float *bias,
                const void *gamma, const void *beta, int m, int n, bool use_fp32)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;

    float local_out = (float)(out[bid * n + tid] + __ldg(&input[bid * n + tid]) + __ldg(&bias[tid]));

    float mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    local_out -= s_mean;
    float variance = blockReduceSum<float>(local_out * local_out);
    if(threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6f);
    __syncthreads();

    out[bid * n + tid] = local_out * s_variance * __ldg(&((float *)gamma)[tid]) + __ldg(&((float *)beta)[tid]);
}

__global__
void add_bias_input_layernorm(
                __half *out, const __half *input, const __half *bias,
                const void *gamma, const void *beta, int m, int n, bool use_fp32)
{
    int tid = threadIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;

    half2 *out_ptr = (half2 *)out;
    const half2 *input_ptr = (const half2 *)input;
    const half2 *bias_ptr  = (const half2 *)bias;

    int id = blockIdx.x * n / 2 + tid;
    float2 local_out_fp2 = __half22float2(
                                           __hadd2(__hadd2(out_ptr[id], __ldg(&input_ptr[id])), __ldg(&bias_ptr[tid])));
    float local_out = local_out_fp2.x + local_out_fp2.y;

    float mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    local_out_fp2.x -= s_mean;
    local_out_fp2.y -= s_mean;
    float variance = local_out_fp2.x * local_out_fp2.x + local_out_fp2.y * local_out_fp2.y;
    variance = blockReduceSum<float>(variance);
    if(threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6f);
    __syncthreads();

    float2 gamma_val, beta_val;
    if(use_fp32)
    {
        gamma_val = __ldg(&((const float2 *)gamma)[tid]);
        beta_val  = __ldg(&((const float2 *)beta)[tid]);
    }
    else
    {
        gamma_val = __half22float2(__ldg(&((const half2 *)gamma)[tid]));
        beta_val  = __half22float2(__ldg(&((const half2 *)beta)[tid]));
    }

    local_out_fp2.x = local_out_fp2.x * s_variance * gamma_val.x + beta_val.x;
    local_out_fp2.y = local_out_fp2.y * s_variance * gamma_val.y + beta_val.y;
    out_ptr[id] = __float22half2_rn(local_out_fp2);
}

__global__
void add_bias_input_layernorm_restore_output(
                const float *out, const float *input, const float *bias,
                const void *gamma, const void *beta, int m, int n, bool use_fp32,
                float *out2, const int *batch_idx, const int *word_idx)
{
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;

    float local_out = (float)(out[blockIdx.x * n + tid] + __ldg(&input[blockIdx.x * n + tid]) + __ldg(&bias[tid]));

    float mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    local_out -= s_mean;
    float variance = blockReduceSum<float>(local_out * local_out);
    if(threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6f);
    __syncthreads();

    int offset = __ldg(&word_idx[blockIdx.x]);
    out2[offset * n + tid] = local_out * s_variance * __ldg(&((float *)gamma)[tid]) + __ldg(&((float *)beta)[tid]);
}

__global__
void add_bias_input_layernorm_restore_output(
                const __half *out, const __half *input, const __half *bias,
                const void *gamma, const void *beta, int m, int n, bool use_fp32,
                __half *out2, const int *batch_idx, const int *word_idx)
{
    int tid = threadIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;

    half2 *out_ptr = (half2 *)out;
    const half2 *input_ptr = (const half2 *)input;
    const half2 *bias_ptr  = (const half2 *)bias;

    int id = blockIdx.x * n / 2 + tid;
    float2 local_out_fp2 = __half22float2(
                                           __hadd2(__hadd2(out_ptr[id], __ldg(&input_ptr[id])), __ldg(&bias_ptr[tid])));
    float local_out = local_out_fp2.x + local_out_fp2.y;

    float mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    local_out_fp2.x -= s_mean;
    local_out_fp2.y -= s_mean;
    float variance = local_out_fp2.x * local_out_fp2.x + local_out_fp2.y * local_out_fp2.y;
    variance = blockReduceSum<float>(variance);
    if(threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6f);
    __syncthreads();

    float2 gamma_val, beta_val;
    if(use_fp32)
    {
        gamma_val = __ldg(&((const float2 *)gamma)[tid]);
        beta_val  = __ldg(&((const float2 *)beta)[tid]);
    }
    else
    {
        gamma_val = __half22float2(__ldg(&((const half2 *)gamma)[tid]));
        beta_val  = __half22float2(__ldg(&((const half2 *)beta)[tid]));
    }

    local_out_fp2.x = local_out_fp2.x * s_variance * gamma_val.x + beta_val.x;
    local_out_fp2.y = local_out_fp2.y * s_variance * gamma_val.y + beta_val.y;

    int offset = __ldg(&word_idx[blockIdx.x]);
    ((half2 *)out2)[offset * n / 2 + tid] = __float22half2_rn(local_out_fp2);
}

}
