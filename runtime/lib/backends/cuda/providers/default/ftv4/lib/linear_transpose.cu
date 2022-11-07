/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v4/includes/gemm.h"
#include "fastertransformer_v4/includes/linear_transpose.h"
#include "fastertransformer_v4/includes/reduce.h"
#include "fastertransformer_v4/includes/utils.h"
using namespace std;

namespace fastertransformerv4 {
template <transposeType transpose_type>
__global__ void add_bias_transpose(const float *input, float *output,
                                   const float *bias) {
  int offset = threadIdx.y * blockDim.x + threadIdx.x;
  int input_offset =
      (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.y * blockDim.x) +
      offset;

  float result = __ldg(&input[input_offset]) + __ldg(&bias[offset]);

  int out_offset =
      transpose3d<transpose_type>(gridDim.y, gridDim.x, blockDim.y, blockIdx.y,
                                  blockIdx.x, threadIdx.y) *
          blockDim.x +
      threadIdx.x;
  output[out_offset] = result;
}

template <transposeType transpose_type>
__global__ void add_bias_transpose(const __half *input, __half *output,
                                   const __half *bias) {
  int offset = threadIdx.y * blockDim.x + threadIdx.x;
  int input_offset =
      (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.y * blockDim.x) +
      offset;

  half2 result = __hadd2(__ldg(&((half2 *)input)[input_offset]),
                         __ldg(&((half2 *)bias)[offset]));

  int out_offset =
      transpose3d<transpose_type>(gridDim.y, gridDim.x, blockDim.y, blockIdx.y,
                                  blockIdx.x, threadIdx.y) *
          blockDim.x +
      threadIdx.x;
  ((half2 *)output)[out_offset] = result;
}

template <transposeType transpose_type>
__global__ void
linear_transpose_bw_dbias_sum(const float *dout, float *transpose_grad_in,
                              float *bias_buf, const int batch_size,
                              const int seq_len, const int head_num,
                              const int size_per_head) {
  int head_id = threadIdx.x / size_per_head;
  int id = threadIdx.x % size_per_head;

  float bias_sum = 0.0f;
  for (int row = blockIdx.x; row < batch_size * seq_len; row += gridDim.x) {
    int batch_id = row / seq_len;
    int seq_id = row % seq_len;

    int grad_out_offset =
        transpose3d<transpose_type>(batch_size, seq_len, head_num, batch_id,
                                    seq_id, head_id) *
            size_per_head +
        id;
    float grad_out = __ldg(&dout[grad_out_offset]);

    int grad_in_offset = row * blockDim.x + threadIdx.x;
    transpose_grad_in[grad_in_offset] = grad_out;

    bias_sum += grad_out;
  }
  bias_buf[blockIdx.x * blockDim.x + threadIdx.x] = bias_sum;
}

template <transposeType transpose_type>
__global__ void
linear_transpose_bw_dbias_sum(const __half *dout, __half *transpose_grad_in,
                              float *bias_buf, const int batch_size,
                              const int seq_len, const int head_num,
                              const int half_size_per_head) {
  int head_id = threadIdx.x / half_size_per_head;
  int id = threadIdx.x % half_size_per_head;

  half2 bias_sum(0.0f, 0.0f);
  for (int row = blockIdx.x; row < batch_size * seq_len; row += gridDim.x) {
    int batch_id = row / seq_len;
    int seq_id = row % seq_len;

    int grad_out_offset =
        transpose3d<transpose_type>(batch_size, seq_len, head_num, batch_id,
                                    seq_id, head_id) *
            half_size_per_head +
        id;
    half2 grad_out = __ldg(&((half2 *)dout)[grad_out_offset]);

    int grad_in_offset = row * blockDim.x + threadIdx.x;
    ((half2 *)transpose_grad_in)[grad_in_offset] = grad_out;

    bias_sum = __hadd2(bias_sum, grad_out);
  }
  ((float2 *)bias_buf)[blockIdx.x * blockDim.x + threadIdx.x] =
      __half22float2(bias_sum);
}

template <typename T>
__global__ void linear_transpose_bw_dbias_reduce(const float *bias_buf,
                                                 T *grad_bias, int hidden_dim,
                                                 int block_count) {
  __shared__ float s_bias[32][32 + 1];

  int warp_id = threadIdx.x >> 5;
  int warp_tid = threadIdx.x & 0x1F;

  int offset = blockIdx.x * 32 + warp_tid;
  const float *bias = bias_buf + offset;

  float sum_bias = 0.0f;
  for (int row = warp_id; row < block_count; row += 32)
    sum_bias += *(bias + row * hidden_dim);

  s_bias[warp_tid][warp_id] = sum_bias;

  __syncthreads();

  float d_bias = warpReduceSum(s_bias[warp_id][warp_tid]);

  if (warp_tid == 0)
    grad_bias[blockIdx.x * 32 + warp_id] = (T)d_bias;
}

template <OperationType OpType>
void LinearTranspose<OpType>::forward(LinearTransposeForwardParam param) {
  const int rows = param.batch_size * seq_len_;
  dense_layer_kernel_launcher(
      param.input, param_.weight, (DataType_ *)param.buf, rows,
      from_hidden_dim_, to_hidden_dim_, CUBLAS_OP_N,
      transposed_weight_ ? CUBLAS_OP_T : CUBLAS_OP_N, (DataType_)1.0f,
      (DataType_)0.0f, param.cublas_handle);

  const int hidden_dim =
      (OpType == OperationType::HALF) ? to_hidden_dim_ / 2 : to_hidden_dim_;
  dim3 grid, block;
  grid.y = param.batch_size, grid.x = seq_len_;
  block.y = head_num_, block.x = hidden_dim / head_num_;
  switch (param.transpose_type) {
  case TRANSPOSE0213:
    add_bias_transpose<TRANSPOSE0213><<<grid, block, 0, param.stream>>>(
        (DataType_ *)param.buf, param.output, param_.bias);
    break;
  case TRANSPOSE2013:
    add_bias_transpose<TRANSPOSE2013><<<grid, block, 0, param.stream>>>(
        (DataType_ *)param.buf, param.output, param_.bias);
    break;
  case TRANSPOSE1203:
    add_bias_transpose<TRANSPOSE1203><<<grid, block, 0, param.stream>>>(
        (DataType_ *)param.buf, param.output, param_.bias);
    break;
  }
}

template <OperationType OpType>
void LinearTranspose<OpType>::backward(LinearTransposeBackwardParam param) {
  float *bias_buf = (float *)param.buf;
  DataType_ *transpose_grad_in_buf =
      (DataType_ *)(bias_buf + block_count_ * to_hidden_dim_);

  const int hidden_dim =
      (OpType == OperationType::HALF) ? to_hidden_dim_ / 2 : to_hidden_dim_;
  dim3 grid, block;

  grid.x = block_count_, block.x = hidden_dim;
  switch (param.transpose_type) {
  case TRANSPOSE0213:
    linear_transpose_bw_dbias_sum<TRANSPOSE0213>
        <<<grid, block, 0, param.stream>>>(
            param.grad_out, transpose_grad_in_buf, bias_buf, param.batch_size,
            seq_len_, head_num_, hidden_dim / head_num_);
    break;
  case TRANSPOSE1203:
    linear_transpose_bw_dbias_sum<TRANSPOSE1203>
        <<<grid, block, 0, param.stream>>>(
            param.grad_out, transpose_grad_in_buf, bias_buf, param.batch_size,
            seq_len_, head_num_, hidden_dim / head_num_);
    break;
  case TRANSPOSE2013:
    linear_transpose_bw_dbias_sum<TRANSPOSE2013>
        <<<grid, block, 0, param.stream>>>(
            param.grad_out, transpose_grad_in_buf, bias_buf, param.batch_size,
            seq_len_, head_num_, hidden_dim / head_num_);
    break;
  }

  grid.x = to_hidden_dim_ / 32, block.x = 1024;
  linear_transpose_bw_dbias_reduce<<<grid, block, 0, param.stream>>>(
      bias_buf, param.grad_bias, to_hidden_dim_, block_count_);

  const int rows = param.batch_size * seq_len_;
  dense_layer_kernel_launcher(
      transpose_grad_in_buf, param_.weight, param.grad_in, rows, to_hidden_dim_,
      from_hidden_dim_, CUBLAS_OP_N,
      transposed_weight_ ? CUBLAS_OP_N : CUBLAS_OP_T, (DataType_)1.0f,
      (DataType_)0.0f, param.cublas_handle);

  if (transposed_weight_)
    dense_layer_kernel_launcher(
        transpose_grad_in_buf, param.input, param.grad_weight, to_hidden_dim_,
        rows, from_hidden_dim_, CUBLAS_OP_T, CUBLAS_OP_N, (DataType_)1.0f,
        (DataType_)0.0f, param.cublas_handle);
  else
    dense_layer_kernel_launcher(
        param.input, transpose_grad_in_buf, param.grad_weight, from_hidden_dim_,
        rows, to_hidden_dim_, CUBLAS_OP_T, CUBLAS_OP_N, (DataType_)1.0f,
        (DataType_)0.0f, param.cublas_handle);
}

template void LinearTranspose<OperationType::FP32>::forward(
    LinearTransposeForwardParam param);
template void LinearTranspose<OperationType::HALF>::forward(
    LinearTransposeForwardParam param);

template void LinearTranspose<OperationType::FP32>::backward(
    LinearTransposeBackwardParam param);
template void LinearTranspose<OperationType::HALF>::backward(
    LinearTransposeBackwardParam param);
} // namespace fastertransformerv4