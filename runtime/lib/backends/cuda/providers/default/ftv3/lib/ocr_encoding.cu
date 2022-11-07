/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v3/includes/common.h"
#include "fastertransformer_v3/includes/operators.cuh"
#include "fastertransformer_v3/includes/utils.h"
#include "fastertransformer_v3/includes/ocr_encoding.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cmath>
using namespace std;

namespace fastertransformerv3
{
template<typename T>
__global__
void pos_encode(const T *src, const T *pe, T *dst)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockIdx.y * gridDim.x * blockDim.x + tid;
    dst[offset] = __ldg(&src[offset]) + __ldg(&pe[tid]);
}

template<>
__global__
void pos_encode(const __half *src, const __half *pe, __half *dst)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockIdx.y * gridDim.x * blockDim.x + tid;
    ((half2 *)dst)[offset] = __hadd2(__ldg(&((half2 *)src)[offset]), __ldg(&((half2 *)pe)[tid]));
}

template<OperationType OpType_>
void OCR_Conformer<OpType_>::infer(OCR_ConformerInferParam infer_param)
{
    int hidden_dim = head_num_ * size_per_head_;
    hidden_dim = (OpType_ == OperationType::HALF) ? (hidden_dim / 2) : hidden_dim;  // for float & half

    dim3 grid(infer_param.seq_len, infer_param.batch_size);
    dim3 block(hidden_dim); //assert block.x <= 1024
    pos_encode<<<grid, block, 0, infer_param.stream>>>(infer_param.input_tensor, param_.pos_encoder_src, infer_param.transformer_output);

    struct ConformerInferParam<DataType_> conformer_infer_param
    {
        infer_param.transformer_output, infer_param.atten_mask, infer_param.transformer_output, infer_param.buf,
                                        infer_param.batch_size, infer_param.seq_len, infer_param.cublas_handle, infer_param.stream
    };

    for(int i = 0; i < layers_; i++)
        conformer_layer_[i]->infer(conformer_infer_param);
}

template void OCR_Conformer<OperationType::FP32>::infer(OCR_ConformerInferParam infer_param);
template void OCR_Conformer<OperationType::HALF>::infer(OCR_ConformerInferParam infer_param);
}
