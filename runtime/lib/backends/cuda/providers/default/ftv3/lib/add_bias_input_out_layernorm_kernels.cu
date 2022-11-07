/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v3/includes/add_bias_input_out_layernorm_kernels.h"
#include "fastertransformer_v3/includes/utils.h"
#include <cuda_fp16.h>

namespace fastertransformerv3
{

__global__
void add_bias_input_out_layernorm(
                float *out, const float *input, const float *bias, float *out2,
                const void *gamma, const void *beta, int m, int n, bool use_fp32)
{
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;

    int offset = blockIdx.x * n + tid;
    float local_out = out[offset] + __ldg(&input[offset]) + __ldg(&bias[tid]);
    out2[offset] = local_out;

    float mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    local_out -= s_mean;
    float variance = blockReduceSum<float>(local_out * local_out);
    if(threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6f);
    __syncthreads();

    out[blockIdx.x * n + tid] = local_out * s_variance * __ldg(&((float *)gamma)[tid]) + __ldg(&((float *)beta)[tid]);
}

__global__
void add_bias_input_out_layernorm(
                __half *out, const __half *input, const __half *bias, __half *out2,
                const void *gamma, const void *beta, int m, int n, bool use_fp32)
{
    int tid = threadIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;

    half2 *out_ptr  = (half2 *)out;
    half2 *out2_ptr = (half2 *)out2;
    const half2 *input_ptr = (const half2 *)input;
    const half2 *bias_ptr  = (const half2 *)bias;

    int id = blockIdx.x * n / 2 + tid;
    half2 temp = __hadd2(__hadd2(out_ptr[id], __ldg(&input_ptr[id])), __ldg(&bias_ptr[tid]));
    out2_ptr[id] = temp;
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
    out_ptr[id] = __float22half2_rn(local_out_fp2);
}
}