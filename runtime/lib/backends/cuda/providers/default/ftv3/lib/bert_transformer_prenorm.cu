/*
* Author: Xiaoying Jia, Changyi Wan
* Project: Faster Transformer Inference
* Department: ByteDance Data-AML
* Email: {jiaxiaoying, wanchangyi}@bytedance.com
*/
#include "fastertransformer_v3/includes/common.h"
#include "fastertransformer_v3/includes/operators.cuh"
#include "fastertransformer_v3/includes/utils.h"
#include "fastertransformer_v3/includes/layernorm_kernels.h"
#include "fastertransformer_v3/includes/add_bias_input_out_layernorm_kernels.h"
#include "fastertransformer_v3/includes/bert_transformer.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cmath>
using namespace std;

namespace fastertransformerv3
{

template<OperationType OpType_>
void BertTransformer<OpType_>::prenorm_bert_infer(BertTransformerInferParam infer_param)
{
    const DataType_* from_tensor = infer_param.input_tensor;    //Todo: remove useless code
    const DataType_* atten_mask  = infer_param.atten_mask;
    DataType_* transformer_out   = infer_param.transformer_output;
    void *buf                    = infer_param.buf;
    const int batch_size         = infer_param.batch_size;
    const int seq_len            = infer_param.seq_len;
    cublasHandle_t cublas_handle = infer_param.cublas_handle;
    cudaStream_t stream          = infer_param.stream;

    int input_tensor_size = batch_size * head_num_ * seq_len * size_per_head_;

    DataType_* attention_buf = (DataType_ *)((uint8_t *)buf + inner_buf_size_);
    DataType_* inner_buf     = (DataType_ *)buf;

    DataType_* query_buf_ = inner_buf + 0 * input_tensor_size;
    DataType_* key_buf_   = inner_buf + 1 * input_tensor_size;
    DataType_* value_buf_ = inner_buf + 2 * input_tensor_size;

    DataType_* attr_out_buf_     = inner_buf + 3 * input_tensor_size;
    DataType_* attr_matmul_buf_  = inner_buf + 1 * input_tensor_size;
    DataType_* inter_matmul_buf_ = inner_buf + 5 * input_tensor_size;

    DataType_* layernorm_tensor  = inner_buf + 3 * input_tensor_size;
    DataType_* middle_tensor     = inner_buf + 4 * input_tensor_size;

    int valid_word_num = batch_size * seq_len;

    int hidden_dim = head_num_ * size_per_head_;
    hidden_dim = (OpType_ == OperationType::HALF) ? (hidden_dim / 2) : hidden_dim;  // for float & half

    ET_Param et_param;
    if(is_remove_padding_)
    {
        et_param.word_idx  = (int *)(inter_matmul_buf_ + 4 * input_tensor_size);
        et_param.batch_idx = et_param.word_idx + batch_size * seq_len;

        build_sequence_length_padding_offset_kernelLauncher(
                        atten_mask, et_param.batch_idx, et_param.word_idx, &valid_word_num,
                        batch_size, seq_len, stream);

        et_param.valid_word_num = valid_word_num;

        input_compress_layernorm_kernel_launcher(
                        layernorm_tensor, from_tensor,
                        param_.attr_output_layernorm_gamma, param_.attr_output_layernorm_beta,
                        valid_word_num, head_num_ * size_per_head_, hidden_dim, stream, use_fp32_,
                        middle_tensor, et_param.batch_idx, et_param.word_idx);

        from_tensor = middle_tensor;        //1. compress from_tensor      -> middle_tensor

        DataType_* tmp  = transformer_out;  //2. compute  transformert_out -> inner_buf
        transformer_out = inner_buf;
        inner_buf = tmp;                    //3. restore  inner_buf        -> from_tensor (real transformer_out)
    }
    else
        input_layernorm_kernel_launcher(
                        layernorm_tensor, from_tensor,
                        param_.attr_output_layernorm_gamma, param_.attr_output_layernorm_beta,
                        valid_word_num, head_num_ * size_per_head_, hidden_dim, stream, use_fp32_);

    int m = valid_word_num;
    int k = head_num_ * size_per_head_;
    int n = k;

    dim3 grid(m);
    dim3 block(hidden_dim); //assert block.x <= 1024

    dense_layer_kernel_launcher(
                    layernorm_tensor, param_.attr_kernel_Q, query_buf_,
                    m, k, n, cublas_handle, stream, param_.cublas_Algo[0]);

    dense_layer_kernel_launcher(
                    layernorm_tensor, param_.attr_kernel_K, key_buf_,
                    m, k, n, cublas_handle, stream, param_.cublas_Algo[0]);

    dense_layer_kernel_launcher(
                    layernorm_tensor, param_.attr_kernel_V, value_buf_,
                    m, k, n, cublas_handle, stream, param_.cublas_Algo[0]);

    attention_layer_->infer(
                    query_buf_, key_buf_, value_buf_, atten_mask, attr_out_buf_, attention_buf,
                    batch_size, seq_len, cublas_handle, stream,
                    et_param);

    dense_layer_kernel_launcher(
                    attr_out_buf_, param_.attr_output_kernel, attr_matmul_buf_,
                    m, k, n, cublas_handle, stream, param_.cublas_Algo[0]);

    add_bias_input_out_layernorm_kernel_launcher(
                    attr_matmul_buf_, from_tensor, param_.attr_output_bias, middle_tensor,
                    param_.output_layernorm_gamma, param_.output_layernorm_beta, m, n, hidden_dim, stream, use_fp32_);

    dense_layer_kernel_launcher(
                    attr_matmul_buf_, param_.inter_kernel, inter_matmul_buf_,
                    m, k, n * 4, cublas_handle, stream, param_.cublas_Algo[1]);

    add_bias_gelu<<<grid, block, 0, stream>>>(
                    inter_matmul_buf_, param_.inter_bias, m, n * 4);

    if(is_remove_padding_)
        cudaMemsetAsync(inner_buf,
                        0, batch_size * seq_len * head_num_ * size_per_head_ * sizeof(DataType_), stream);

    dense_layer_kernel_launcher(
                    inter_matmul_buf_, param_.output_kernel, transformer_out,
                    m, k * 4, n, cublas_handle, stream, param_.cublas_Algo[2]);

    if(is_remove_padding_)
        add_bias_input_restore_output<<<grid, block, 0, stream>>>(
                        transformer_out, middle_tensor, param_.output_bias, m, n,
                        inner_buf, et_param.batch_idx, et_param.word_idx);
    else
        add_bias_input<<<grid, block, 0, stream>>>(
                        transformer_out, middle_tensor, param_.output_bias, m, n);
}

template void BertTransformer<OperationType::FP32>::prenorm_bert_infer(BertTransformerInferParam infer_param);
template void BertTransformer<OperationType::HALF>::prenorm_bert_infer(BertTransformerInferParam infer_param);
}
