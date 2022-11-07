/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v4/includes/gemm.h"
#include "fastertransformer_v4/includes/linear.h"
#include "fastertransformer_v4/includes/reduce.h"
#include "fastertransformer_v4/includes/utils.h"
using namespace std;

namespace fastertransformerv4 {
template <typename T>
__global__ void add_bias_gelu_dropout(T *input, const T *bias, const int M,
                                      const int N, bool act_gelu,
                                      float dropout_rate, T *bias_out,
                                      uint8_t *dropout_mask, const float ratio,
                                      const int seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = tid * 4;

  float4 in = load_vector(input + offset);
  float4 bias4 = load_vector(bias + threadIdx.x * 4);

  in.x += bias4.x;
  in.y += bias4.y;
  in.z += bias4.z;
  in.w += bias4.w;

  if (act_gelu) {
    store_vector(bias_out + offset, in);
    in.x = gelu_fw(in.x);
    in.y = gelu_fw(in.y);
    in.z = gelu_fw(in.z);
    in.w = gelu_fw(in.w);
  }
  if (dropout_rate > 0.0f)
    in = dropout_fw(in, ratio, seed, tid, (uchar4 *)dropout_mask);

  store_vector(input + offset, in);
}

template <typename T>
__global__ void
linear_bw_dropout_gelu_dbias_sum(const T *dout, float *bias_buf, int hidden_dim,
                                 int rows, bool act_gelu, T *bias_out,
                                 uint8_t *dropout_mask, float scale) {
  const T *dout_buf = dout + threadIdx.x * 4;

  float4 bias_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    int offset = row * hidden_dim;
    float4 dout4 = load_vector(dout_buf + offset);

    if (scale > 1.0f) // apply_dropout
      dout4 = dropout_bw(dout4, scale, offset / 4 + threadIdx.x,
                         (uchar4 *)dropout_mask);

    T *bias_out_buf = bias_out + offset + threadIdx.x * 4;
    if (act_gelu) {
      float4 bias_out4 = load_vector(bias_out_buf);
      dout4.x = gelu_bw(dout4.x, bias_out4.x);
      dout4.y = gelu_bw(dout4.y, bias_out4.y);
      dout4.z = gelu_bw(dout4.z, bias_out4.z);
      dout4.w = gelu_bw(dout4.w, bias_out4.w);
    }

    if (scale > 1.0f || act_gelu)
      store_vector(bias_out_buf, dout4);

    bias_sum.x += dout4.x;
    bias_sum.y += dout4.y;
    bias_sum.z += dout4.z;
    bias_sum.w += dout4.w;
  }

  store_vector(bias_buf + blockIdx.x * hidden_dim + threadIdx.x * 4, bias_sum);
}

template <typename T>
__global__ void linear_bw_dbias_reduce(const float *bias_buf, T *grad_bias,
                                       int hidden_dim, int block_count) {
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
void Linear<OpType>::forward(LinearForwardParam param) {
  dense_layer_kernel_launcher(
      param.input, param_.weight, param.output, param.rows, K_, N_, CUBLAS_OP_N,
      transposed_weight_ ? CUBLAS_OP_T : CUBLAS_OP_N, (DataType_)1.0f,
      (DataType_)0.0f, param.cublas_handle);

  const int seed = generate_random_seed();
  add_bias_gelu_dropout<<<param.rows, N_ / 4, 0, param.stream>>>(
      param.output, param_.bias, param.rows, N_, act_gelu_, dropout_rate_,
      param.bias_out, param.dropout_mask, dropout_rate_, seed);
}

template <OperationType OpType>
void Linear<OpType>::backward(LinearBackwardParam param) {
  float *bias_buf = (float *)param.buf;

  dim3 grid, block;
  grid.x = block_count_, block.x = N_ / 4;
  linear_bw_dropout_gelu_dbias_sum<<<grid, block, 0, param.stream>>>(
      param.grad_out, bias_buf, N_, param.rows, act_gelu_, param.bias_out,
      param.dropout_mask, 1.0f / (1.0f - dropout_rate_));

  grid.x = N_ / 32, block.x = 1024;
  linear_bw_dbias_reduce<<<grid, block, 0, param.stream>>>(
      bias_buf, param.grad_bias, N_, block_count_);

  const DataType_ *grad_ptr =
      (act_gelu_ || dropout_rate_ > 0.0f) ? param.bias_out : param.grad_out;
  dense_layer_kernel_launcher(
      grad_ptr, param_.weight, param.grad_in, param.rows, N_, K_, CUBLAS_OP_N,
      transposed_weight_ ? CUBLAS_OP_N : CUBLAS_OP_T, (DataType_)1.0f,
      (DataType_)0.0f, param.cublas_handle);

  if (transposed_weight_)
    dense_layer_kernel_launcher(grad_ptr, param.input, param.grad_weight, N_,
                                param.rows, K_, CUBLAS_OP_T, CUBLAS_OP_N,
                                (DataType_)1.0f, (DataType_)0.0f,
                                param.cublas_handle);
  else
    dense_layer_kernel_launcher(param.input, grad_ptr, param.grad_weight, K_,
                                param.rows, N_, CUBLAS_OP_T, CUBLAS_OP_N,
                                (DataType_)1.0f, (DataType_)0.0f,
                                param.cublas_handle);
}

template void Linear<OperationType::FP32>::forward(LinearForwardParam param);
template void Linear<OperationType::HALF>::forward(LinearForwardParam param);

template void Linear<OperationType::FP32>::backward(LinearBackwardParam param);
template void Linear<OperationType::HALF>::backward(LinearBackwardParam param);
} // namespace fastertransformerv4