/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v3/includes/utils.h"
#include "fastertransformer_v3/includes/convolution.h"
#include <cuda_fp16.h>
using namespace std;

namespace fastertransformerv3
{
__global__
void add_bias_glu(const float *input, const float *bias, float *output, const int M, const int N)
{
    int row_offset = blockIdx.x * N;
    int tid = threadIdx.x;

    float front = __ldg(&input[row_offset + tid]) + __ldg(&bias[tid]);
    float back  = __ldg(&input[row_offset + (tid + blockDim.x)]) + __ldg(&bias[tid + blockDim.x]);

    output[row_offset / 2 + tid] = front / (1.0f + __expf(-back));
}

__global__
void add_bias_glu(const __half *input, const __half *bias, __half *output, const int M, const int N)
{
    const half2 *input_ptr = (half2 *)input;
    const half2 *bias_ptr = (const half2 *)bias;
    half2 *output_ptr = (half2 *)output;

    int row_offset = blockIdx.x * N / 2;
    int tid = threadIdx.x;

    float2 front = __half22float2(__hadd2(__ldg(&input_ptr[row_offset + tid]), __ldg(&bias_ptr[tid])));
    float2 back  = __half22float2(__hadd2(__ldg(&input_ptr[row_offset + (tid + blockDim.x)]), __ldg(&bias_ptr[tid + blockDim.x])));

    float2 result;
    result.x = (front.x) / (1.0f + __expf(-back.x));
    result.y = (front.y) / (1.0f + __expf(-back.y));

    output_ptr[row_offset / 2 + tid] = __float22half2_rn(result);
}

__global__
void transpose_to_NCL(const float *input, float *output, const int seq_len, const int hidden_dim)
{
    __shared__ float s_in[32][32 + 1];
    if(blockIdx.y * 32 + threadIdx.y < seq_len)
    {
        int offset = (blockIdx.z * seq_len * hidden_dim) + (blockIdx.y * 32 * hidden_dim) + (blockIdx.x * 32) + (threadIdx.y * hidden_dim);
        s_in[threadIdx.y][threadIdx.x] = __ldg(&input[offset + threadIdx.x]);
    }
    __syncthreads();
    if(blockIdx.y * 32 + threadIdx.x < seq_len)
    {
        int offset = (blockIdx.z * seq_len * hidden_dim) + (blockIdx.x * 32 * seq_len) + (blockIdx.y * 32) + (threadIdx.y * seq_len);
        output[offset + threadIdx.x] = s_in[threadIdx.x][threadIdx.y];
    }
}

__global__
void transpose_to_NCL(const __half *input, __half *output, const int seq_len, const int hidden_dim)
{
    __shared__ float s_in[32][32 + 1]; //todo: use half2
    if(blockIdx.y * 32 + threadIdx.y < seq_len)
    {
        int offset = (blockIdx.z * seq_len * hidden_dim) + (blockIdx.y * 32 * hidden_dim) + (blockIdx.x * 32) + (threadIdx.y * hidden_dim);
        s_in[threadIdx.y][threadIdx.x] = (float)__ldg(&input[offset + threadIdx.x]);
    }
    __syncthreads();
    if(blockIdx.y * 32 + threadIdx.x < seq_len)
    {
        int offset = (blockIdx.z * seq_len * hidden_dim) + (blockIdx.x * 32 * seq_len) + (blockIdx.y * 32) + (threadIdx.y * seq_len);
        output[offset + threadIdx.x] = (__half)s_in[threadIdx.x][threadIdx.y];
    }
}

__global__
void transpose_to_NLC(const float *input, float *output, const int hidden_dim, const int seq_len)
{
    __shared__ float s_in[32][32 + 1];
    if(blockIdx.x * 32 + threadIdx.x < seq_len)
    {
        int offset = (blockIdx.z * hidden_dim * seq_len) + (blockIdx.y * 32 * seq_len) + (blockIdx.x * 32) + (threadIdx.y * seq_len);
        s_in[threadIdx.y][threadIdx.x] = __ldg(&input[offset + threadIdx.x]);
    }
    __syncthreads();
    if(blockIdx.x * 32 + threadIdx.y < seq_len)
    {
        int offset = (blockIdx.z * seq_len * hidden_dim) + (blockIdx.x * 32 * hidden_dim) + (blockIdx.y * 32) + (threadIdx.y * hidden_dim);
        output[offset + threadIdx.x] = s_in[threadIdx.x][threadIdx.y];
    }
}

__global__
void transpose_to_NLC(const __half *input, __half *output, const int hidden_dim, const int seq_len)
{
    __shared__ float s_in[32][32 + 1]; //todo: use half2
    if(blockIdx.x * 32 + threadIdx.x < seq_len)
    {
        int offset = (blockIdx.z * hidden_dim * seq_len) + (blockIdx.y * 32 * seq_len) + (blockIdx.x * 32) + (threadIdx.y * seq_len);
        s_in[threadIdx.y][threadIdx.x] = (float)__ldg(&input[offset + threadIdx.x]);
    }
    __syncthreads();
    if(blockIdx.x * 32 + threadIdx.y < seq_len)
    {
        int offset = (blockIdx.z * seq_len * hidden_dim) + (blockIdx.x * 32 * hidden_dim) + (blockIdx.y * 32) + (threadIdx.y * hidden_dim);
        output[offset + threadIdx.x] = (__half)s_in[threadIdx.x][threadIdx.y];
    }
}

__global__
void depthwise_conv(const float *input, const float *conv_kernel, float *output, int seq_len, int N)
{
    __shared__ float s_conv_weight[31];

    int channel_id = blockIdx.x;
    int batch_id = blockIdx.y;

    if(threadIdx.x < 31)
        s_conv_weight[threadIdx.x] = __ldg(&conv_kernel[channel_id * 31 + threadIdx.x]);
    __syncthreads();

    if(threadIdx.x < seq_len)
    {
        int offset = (batch_id * N + channel_id) * seq_len + threadIdx.x;

        float sum = 0.0f;
        for(int i = -15; i <= 15; i++)
        {
            int pos = threadIdx.x + i;
            float in = (pos >= 0 && pos < seq_len) ? __ldg(&input[offset + i]) : 0.0f;
            sum += s_conv_weight[15 + i] * in;
        }

        output[offset] = sum;
    }
}

__global__
void depthwise_conv(const __half *input, const __half *conv_kernel, __half *output, int seq_len, int N)
{
    __shared__ float s_conv_weight[31];

    int channel_id = blockIdx.x;
    int batch_id = blockIdx.y;

    if(threadIdx.x < 31)
        s_conv_weight[threadIdx.x] = (float)__ldg(&conv_kernel[channel_id * 31 + threadIdx.x]);
    __syncthreads();

    if(threadIdx.x < seq_len)
    {
        int offset = (batch_id * N + channel_id) * seq_len + threadIdx.x;

        float sum = 0.0f;
        for(int i = -15; i <= 15; i++)
        {
            int pos = threadIdx.x + i;
            float in = (pos >= 0 && pos < seq_len) ? (float)__ldg(&input[offset + i]) : 0.0f;
            sum += s_conv_weight[15 + i] * in;
        }

        output[offset] = (__half)sum;
    }
}

__global__
void add_bias_batchnorm_swish(float *output, const float *bias, const float *mean, const float *var, const void *gamma, const void *beta, int m, int n, int hidden_dim, bool use_fp32)
{
    int tid = threadIdx.x;
    int offset = blockIdx.x * n + tid;
    float local_out = output[offset] + __ldg(&bias[tid]) - __ldg(&mean[tid]);
    float variance = rsqrtf(__ldg(&var[tid]) + 1e-5f);
    float result = local_out * variance * __ldg(&((float *)gamma)[tid]) + __ldg(&((float *)beta)[tid]);
    output[offset] = swish(result);
}

__global__
void add_bias_batchnorm_swish(__half *output, const __half *bias, const __half *mean, const __half *var, const void *gamma, const void *beta, int m, int n, int hidden_dim, bool use_fp32)
{
    int tid = threadIdx.x;
    int offset = blockIdx.x * n / 2 + tid;

    half2 *out_ptr = (half2 *)output;
    const half2 *bias_ptr = (const half2 *)bias;
    const half2 *mean_ptr = (const half2 *)mean;
    const half2 *var_ptr  = (const half2 *)var;

    float2 local_out_fp2 = __half22float2(__hsub2(__hadd2(out_ptr[offset], __ldg(&bias_ptr[tid])), __ldg(&mean_ptr[tid])));

    float2 variance = __half22float2(__ldg(&var_ptr[tid]));
    variance.x = rsqrtf(variance.x + 1e-5f);
    variance.y = rsqrtf(variance.y + 1e-5f);

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

    local_out_fp2.x = local_out_fp2.x * variance.x * gamma_val.x + beta_val.x;
    local_out_fp2.y = local_out_fp2.y * variance.y * gamma_val.y + beta_val.y;
    half2 result = __float22half2_rn(local_out_fp2);
    out_ptr[offset] = swish(result);
}

}