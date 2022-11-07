/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v3/includes/common.h"
#include "fastertransformer_v3/includes/operators.cuh"
#include "fastertransformer_v3/includes/utils.h"
#include "fastertransformer_v3/includes/fusion_score.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cmath>
using namespace std;

namespace fastertransformerv3
{

template <typename T>
__global__
void transpose_dim01(
                const T *input, T *output, const int dim0, const int dim1, const int num_units)
{
    int bid  = blockIdx.x;
    int idx0 = bid / dim1;
    int idx1 = bid % dim1;

    int source_offset = (idx0 * dim1 + idx1) * num_units;
    int target_offset = (idx1 * dim0 + idx0) * num_units;

    for(int tid = threadIdx.x; tid < num_units; tid += blockDim.x)
        output[target_offset + tid] = __ldg(&input[source_offset + tid]);
}

template <ActType act, typename T>
__global__
void matrix_mul_vector_add_bias_act(
                const T *input, const T *weight, const T *bias,
                T *output, const int M, const int K)
{
    int row_offset = blockIdx.x * K;

    T sum = (T) 0.0f;
    for(int tid = threadIdx.x; tid < K; tid += 32)
        sum += input[row_offset + tid] * weight[tid];

    sum = warpReduceSum(sum);

    if(threadIdx.x == 0)
    {
        T out = sum + __ldg(&bias[0]);
        output[blockIdx.x] = act_fun<act>(out);
    }
}

template<OperationType OpType_>
void FusionScore<OpType_>::fused_infer(
                const DataType_* query, const DataType_* key, const DataType_* value, const DataType_* key_padding_mask,
                DataType_* score, void *buf,
                const int batch_size, const int from_seq_len, const int to_seq_len, cublasHandle_t cublas_handle, cudaStream_t stream)
{
    DataType_* key_T = (DataType_ *)buf;
    DataType_* attention_output = (DataType_ *)key_T   + key_T_size_;
    DataType_* mid_out = (DataType_ *)attention_output + attention_output_size_;
    DataType_* multi_head_attention_buf = (DataType_ *)mid_out + mid_out_size_;

    dim3 grid(batch_size * to_seq_len);
    dim3 block(hidden_dim_); //assert block.x <= 1024

    transpose_dim01<<<grid, block, 0, stream>>>(key, key_T, batch_size, to_seq_len, hidden_dim_);

    multi_head_attention_layer_->infer(
                    query, key_T, key_T, key_padding_mask,
                    attention_output, multi_head_attention_buf,
                    batch_size, from_seq_len, to_seq_len, cublas_handle, stream);

    dense_layer_kernel_launcher(
                    attention_output, param_.linear1_weight, mid_out,
                    batch_size * from_seq_len, hidden_dim_, fc_hidden_size1_, cublas_handle, stream, param_.cublas_Algo[0]);

    grid.x = batch_size * from_seq_len;
    block.x = fc_hidden_size1_; //assert block.x <= 1024

    add_bias_act<ActType::Relu, DataType_><<<grid, block, 0, stream>>>(mid_out, param_.linear1_bias, batch_size * from_seq_len, fc_hidden_size1_);

    // dense_layer_kernel_launcher(
    //     mid_out, param_.linear2_weight, score,
    //     batch_size * from_seq_len, fc_hidden_size1_, fc_hidden_size2_, cublas_handle, stream, param_.cublas_Algo[1]);

    // grid.x = batch_size * from_seq_len;
    // block.x = 1; //assert block.x <= 1024

    // add_bias_act<ActType::Sigmoid, DataType_><<<grid, block, 0, stream>>>(score, param_.linear2_bias, batch_size * from_seq_len, fc_hidden_size2_);

    grid.x = batch_size * from_seq_len;
    block.x = 32; // fc_hidden_size1_ -> warpReduce
    matrix_mul_vector_add_bias_act<ActType::Sigmoid, DataType_><<<grid, block, 0, stream>>>(
                    mid_out, param_.linear2_weight, param_.linear2_bias, score,
                    batch_size * from_seq_len, fc_hidden_size1_);
}

template void FusionScore<OperationType::FP32>::fused_infer(
                const float *query, const float *key, const float *value, const float *key_padding_mask,
                float *attn_output, void *buf,
                const int batch_size, const int from_seq_len, const int to_seq_len, cublasHandle_t cublas_handle, cudaStream_t stream);

template void FusionScore<OperationType::HALF>::fused_infer(
                const __half *query, const __half *key, const __half *value, const __half *key_padding_mask,
                __half *attn_output, void *buf,
                const int batch_size, const int from_seq_len, const int to_seq_len, cublasHandle_t cublas_handle, cudaStream_t stream);
}
