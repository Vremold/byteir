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
#include "fastertransformer_v3/includes/add_bias_half_input_out_layernorm_kernels.h"
#include "fastertransformer_v3/includes/add_bias_input_out_layernorm_kernels.h"
#include "fastertransformer_v3/includes/convolution.h"
#include "fastertransformer_v3/includes/conformer.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cmath>
using namespace std;

namespace fastertransformerv3
{

template<OperationType OpType_>
void Conformer<OpType_>::infer(ConformerInferParam infer_param)
{
    const DataType_* from_tensor = infer_param.input_tensor;    //Todo: remove useless code
    const DataType_* atten_mask  = infer_param.atten_mask;
    DataType_* transformer_out   = infer_param.transformer_output;
    void *buf                    = infer_param.buf;
    const int batch_size         = infer_param.batch_size;
    const int seq_len            = infer_param.seq_len;
    cublasHandle_t cublas_handle = infer_param.cublas_handle;
    cudaStream_t stream          = infer_param.stream;

    int input_tensor_size = batch_size * head_num_ * seq_len * size_per_head_; //todo: set max_seq_len

    DataType_* attention_buf = (DataType_ *)((uint8_t *)buf + inner_buf_size_);
    DataType_* inner_buf     = (DataType_ *)buf;

    DataType_* layernorm_tensor  = inner_buf + 0 * input_tensor_size;

    DataType_* ffn1_inter_matmul_buf_ = inner_buf + 1 * input_tensor_size;
    DataType_* ffn1_out_matmul_buf_   = inner_buf + 2 * input_tensor_size;

    DataType_* new_from_tensor        = inner_buf + 3 * input_tensor_size;

    DataType_* query_buf_ = inner_buf + 4 * input_tensor_size;
    DataType_* key_buf_   = inner_buf + 5 * input_tensor_size;
    DataType_* value_buf_ = inner_buf + 6 * input_tensor_size;

    DataType_* attr_out_buf_     = inner_buf + 7 * input_tensor_size;
    DataType_* attr_matmul_buf_  = inner_buf + 8 * input_tensor_size;
    DataType_* inter_matmul_buf_ = inner_buf + 9 * input_tensor_size;
    DataType_* middle_tensor     = inner_buf + 10 * input_tensor_size;

    DataType_* conv_matmul_buf_  = inner_buf + 11 * input_tensor_size; // *2
    DataType_* glu_out_buf_      = inner_buf + 13 * input_tensor_size;

    DataType_* transpose_buf_    = inner_buf + 14 * input_tensor_size;

    int valid_word_num = batch_size * seq_len;

    int hidden_dim = head_num_ * size_per_head_;
    hidden_dim = (OpType_ == OperationType::HALF) ? (hidden_dim / 2) : hidden_dim;  // for float & half

    ET_Param et_param;
    if(is_remove_padding_)
    {
        et_param.word_idx  = (int *)(inner_buf + 15 * input_tensor_size);
        et_param.batch_idx = et_param.word_idx + batch_size * seq_len;

        build_sequence_length_padding_offset_kernelLauncher(
                        atten_mask, et_param.batch_idx, et_param.word_idx, &valid_word_num,
                        batch_size, seq_len, stream);

        et_param.valid_word_num = valid_word_num;

        input_compress_layernorm_kernel_launcher(
                        layernorm_tensor, from_tensor,
                        param_.ffn1_layernorm_gamma, param_.ffn1_layernorm_beta,
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
                        param_.ffn1_layernorm_gamma, param_.ffn1_layernorm_beta,
                        valid_word_num, head_num_ * size_per_head_, hidden_dim, stream, use_fp32_);

    int m = valid_word_num;
    int k = head_num_ * size_per_head_;
    int n = k;

    dim3 grid(m);
    dim3 block(hidden_dim); //assert block.x <= 1024

    // FFN1 (pre-layernorm)
    dense_layer_kernel_launcher(
                    layernorm_tensor, param_.ffn1_inter_kernel, ffn1_inter_matmul_buf_,
                    m, k, n, cublas_handle, stream, param_.cublas_Algo[1]);

    add_bias_swish<<<grid, block, 0, stream>>>(
                    ffn1_inter_matmul_buf_, param_.ffn1_inter_bias, m, n);

    dense_layer_kernel_launcher(
                    ffn1_inter_matmul_buf_, param_.ffn1_output_kernel, ffn1_out_matmul_buf_,
                    m, k, n, cublas_handle, stream, param_.cublas_Algo[2]);

    add_bias_half_input_out_layernorm_kernel_launcher(
                    ffn1_out_matmul_buf_, from_tensor, param_.ffn1_output_bias, new_from_tensor,
                    param_.attr_output_layernorm_gamma, param_.attr_output_layernorm_beta, m, n, hidden_dim, stream, use_fp32_);

    //Multi-Head Self Attention
    dense_layer_kernel_launcher(
                    ffn1_out_matmul_buf_, param_.attr_kernel_Q, query_buf_,
                    m, k, n, cublas_handle, stream, param_.cublas_Algo[0]);

    dense_layer_kernel_launcher(
                    ffn1_out_matmul_buf_, param_.attr_kernel_K, key_buf_,
                    m, k, n, cublas_handle, stream, param_.cublas_Algo[0]);

    dense_layer_kernel_launcher(
                    ffn1_out_matmul_buf_, param_.attr_kernel_V, value_buf_,
                    m, k, n, cublas_handle, stream, param_.cublas_Algo[0]);

    attention_layer_->infer(
                    query_buf_, key_buf_, value_buf_, atten_mask, attr_out_buf_, attention_buf,
                    batch_size, seq_len, cublas_handle, stream,
                    et_param);

    dense_layer_kernel_launcher(
                    attr_out_buf_, param_.attr_output_kernel, attr_matmul_buf_,
                    m, k, n, cublas_handle, stream, param_.cublas_Algo[0]);

    add_bias_input_out_layernorm_kernel_launcher(
                    attr_matmul_buf_, new_from_tensor, param_.attr_output_bias, middle_tensor,
                    param_.conv_layernorm_gamma, param_.conv_layernorm_beta, m, n, hidden_dim, stream, use_fp32_);

    // Convolution
    dense_layer_kernel_launcher(
                    attr_matmul_buf_, param_.pointwise_conv_kernel_1, conv_matmul_buf_,
                    m, k, n * 2, cublas_handle, stream); //param_.cublas_Algo[5]

    add_bias_glu<<<grid, block, 0, stream>>>(
                    conv_matmul_buf_, param_.pointwise_conv_bias_1, glu_out_buf_, m, n * 2);

    transpose_to_NCL<<<dim3(n / 32, (seq_len + 31) / 32, batch_size), dim3(32, 32), 0, stream>>>(
                    glu_out_buf_, transpose_buf_, seq_len, n); //[N,L,C] -> [N,C,L] (assert L >= 32, padding for L)

    // add_bias_glu_transpose_dim12<<<dim3(n / 32, (seq_len + 31) / 32, batch_size), dim3(32, 32), 0, stream>>>(
    //    conv_matmul_buf_, param_.pointwise_conv_bias_1, transpose_buf_, seq_len, n);

    depthwise_conv<<<dim3(n, batch_size), max(seq_len, 32), 0, stream>>>(
                    transpose_buf_, param_.depthwise_conv_kernel, conv_matmul_buf_, seq_len, n); //todo: _normal & _et

    // add_bias_layernorm_swish_transpose_dim12<<<dim3((seq_len + 31) / 32, n / 32, batch_size), dim3(32, 32), 0, stream>>>(
    //     transpose_buf_, param_.depthwise_conv_bias,
    //     param_.batchnorm_mean, param_.batchnorm_var, param_.batchnorm_gamma, param_.batchnorm_beta,
    //     glu_out_buf_, n, seq_len); //[N,C,L] -> [N,L,C] remove padding of L

    // transpose_dim12_add_bias_layernorm_swish<<<dim3((seq_len + 31) / 32, n / 32, batch_size), dim3(32, 32), 0, stream>>>(
    //     transpose_buf_, param_.depthwise_conv_bias,
    //     param_.batchnorm_mean, param_.batchnorm_var, param_.batchnorm_gamma, param_.batchnorm_beta,
    //     glu_out_buf_, n, seq_len); //[N,C,L] -> [N,L,C] remove padding of L

    transpose_to_NLC<<<dim3((seq_len + 31) / 32, n / 32, batch_size), dim3(32, 32), 0, stream>>>(
                    conv_matmul_buf_, glu_out_buf_, n, seq_len); //[N,C,L] -> [N,L,C]

    add_bias_batchnorm_swish<<<grid, block, 0, stream>>>(
                    glu_out_buf_, param_.depthwise_conv_bias,
                    param_.batchnorm_mean, param_.batchnorm_var, param_.batchnorm_gamma, param_.batchnorm_beta,
                    m, n, hidden_dim, use_fp32_);

    dense_layer_kernel_launcher(
                    glu_out_buf_, param_.pointwise_conv_kernel_2, conv_matmul_buf_,
                    m, k, n, cublas_handle, stream); //param_.cublas_Algo[5]

    add_bias_input_out_layernorm_kernel_launcher(
                    conv_matmul_buf_, middle_tensor, param_.pointwise_conv_bias_2, new_from_tensor,
                    param_.output_layernorm_gamma, param_.output_layernorm_beta, m, n, hidden_dim, stream, use_fp32_);

    // FFN2 (pre-layernorm)
    dense_layer_kernel_launcher(
                    conv_matmul_buf_, param_.inter_kernel, inter_matmul_buf_,
                    m, k, n, cublas_handle, stream, param_.cublas_Algo[1]);

    add_bias_swish<<<grid, block, 0, stream>>>(
                    inter_matmul_buf_, param_.inter_bias, m, n);

    if(is_remove_padding_)
        cudaMemsetAsync(inner_buf,
                        0, batch_size * seq_len * head_num_ * size_per_head_ * sizeof(DataType_), stream);

    dense_layer_kernel_launcher(
                    inter_matmul_buf_, param_.output_kernel, transformer_out,
                    m, k, n, cublas_handle, stream, param_.cublas_Algo[2]);

    if(is_remove_padding_)
        add_bias_half_input_layernorm_restore_output_kernel_launcher(
                        transformer_out, new_from_tensor, param_.output_bias,
                        param_.last_layernorm_gamma, param_.last_layernorm_beta, m, n, hidden_dim, stream, use_fp32_,
                        inner_buf, et_param.batch_idx, et_param.word_idx);
    else
        add_bias_half_input_layernorm_kernel_launcher(
                        transformer_out, new_from_tensor, param_.output_bias,
                        param_.last_layernorm_gamma, param_.last_layernorm_beta, m, n, hidden_dim, stream, use_fp32_);
}

template void Conformer<OperationType::FP32>::infer(ConformerInferParam infer_param);
template void Conformer<OperationType::HALF>::infer(ConformerInferParam infer_param);
}
