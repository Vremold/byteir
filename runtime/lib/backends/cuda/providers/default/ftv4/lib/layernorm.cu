/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v4/includes/layernorm.h"
#include "fastertransformer_v4/includes/reduce.h"
#include "fastertransformer_v4/includes/utils.h"
using namespace std;

namespace fastertransformerv4 {
#define LN_EPSILON 1e-6f
#define WARP_SIZE 32

template <typename T>
__global__ void layernorm_fw(const T *input, const T *gamma, const T *beta,
                             const T *residual, T *mean_, T *var_rsqrt_,
                             T *layernorm_out, T *input_add_residual,
                             float r_hidden_dim) {
  int offset = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

  float4 in = load_vector(input + offset);
  if (residual) {
    float4 res4 = load_vector(residual + offset);
    in.x += res4.x;
    in.y += res4.y;
    in.z += res4.z;
    in.w += res4.w;
    store_vector(input_add_residual + offset, in);
  }

  __shared__ float s_mean, s_var_rsqrt;
  float mean, var_rsqrt;
  float2 sum2;
  sum2.x = in.x + in.y + in.z + in.w;
  sum2.y = in.x * in.x + in.y * in.y + in.z * in.z + in.w * in.w;
  sum2 = blockReduceSum_2(sum2);
  if (threadIdx.x == 0) {
    mean = sum2.x * r_hidden_dim;
    s_mean = mean;
    float var = sum2.y * r_hidden_dim - mean * mean;
    s_var_rsqrt = rsqrtf(var > LN_EPSILON ? var : LN_EPSILON);
  }
  __syncthreads();
  mean = s_mean, var_rsqrt = s_var_rsqrt;

  // float sum = in.x + in.y + in.z + in.w;
  // sum = blockReduceSum<float>(sum);
  // if(threadIdx.x == 0)
  //     s_mean = sum * r_hidden_dim;
  // __syncthreads();
  // mean = s_mean;

  in.x -= mean;
  in.y -= mean;
  in.z -= mean;
  in.w -= mean;

  // float variance = in.x * in.x + in.y * in.y + in.z * in.z + in.w * in.w;
  // variance = blockReduceSum<float>(variance);
  // if(threadIdx.x == 0)
  //     s_var_rsqrt = rsqrtf(variance * r_hidden_dim + 1e-6f);
  // __syncthreads();
  // var_rsqrt = s_var_rsqrt;

  float4 gamma4 = load_vector(gamma + threadIdx.x * 4);
  float4 beta4 = load_vector(beta + threadIdx.x * 4);
  float4 out;
  out.x = in.x * var_rsqrt * gamma4.x + beta4.x;
  out.y = in.y * var_rsqrt * gamma4.y + beta4.y;
  out.z = in.z * var_rsqrt * gamma4.z + beta4.z;
  out.w = in.w * var_rsqrt * gamma4.w + beta4.w;

  store_vector(layernorm_out + offset, out);
  if (threadIdx.x == 0) {
    mean_[blockIdx.x] = (T)mean;
    var_rsqrt_[blockIdx.x] = (T)var_rsqrt;
  }
}

template <typename T>
__global__ void layernorm_fw_mini_dim(const T *input, const T *gamma_ptr,
                                      const T *beta_ptr, const T *residual,
                                      T *mean_, T *var_rsqrt_, T *layernorm_out,
                                      T *input_add_residual, float r_hidden_dim,
                                      int rows, int hidden_dim) {
  if ((blockIdx.x * blockDim.y + threadIdx.y) >= rows)
    return;
  const int max_warp_per_row = 4;

  register float s_in[max_warp_per_row];
  register float s_in2[max_warp_per_row];
  int offset = (blockIdx.x * blockDim.y + threadIdx.y) * hidden_dim;
  float2 sum2;
  sum2.x = 0.0f;
  sum2.y = 0.0f;
  for (int idx = threadIdx.x, warp_idx = 0; idx < hidden_dim;
       idx += WARP_SIZE, warp_idx++) {
    s_in[warp_idx] = (float)input[offset + idx];
    if (residual) {
      s_in[warp_idx] += (float)residual[offset + idx];
      input_add_residual[offset + idx] = (T)s_in[warp_idx];
    }
    s_in2[warp_idx] = (float)(s_in[warp_idx] * s_in[warp_idx]);
    sum2.x += s_in[warp_idx];
    sum2.y += s_in2[warp_idx];
  }
  __syncwarp();
  sum2 = warpReduceSum_2(sum2);

  float mean = sum2.x * r_hidden_dim;
  float var = sum2.y * r_hidden_dim - mean * mean;
  float var_rsqrt = rsqrtf(var > LN_EPSILON ? var : LN_EPSILON);
#pragma unroll
  for (int idx = threadIdx.x, warp_idx = 0; idx < hidden_dim;
       idx += WARP_SIZE, warp_idx++) {
    register float in = s_in[warp_idx];
    float gamma = gamma_ptr[idx];
    float beta = beta_ptr[idx];
    in -= mean;
    layernorm_out[offset + idx] = in * var_rsqrt * gamma + beta;
  }
  if (threadIdx.x == 0) {
    mean_[blockIdx.x * blockDim.y + threadIdx.y] = (T)mean;
    var_rsqrt_[blockIdx.x * blockDim.y + threadIdx.y] = (T)var_rsqrt;
  }
}

template <typename T>
__global__ void
layernorm_bw_dgamma_dbeta_sum(const T *dout, const T *input, const T *mean_,
                              const T *var_rsqrt_, float *gamma_buf,
                              float *beta_buf, int hidden_dim, int rows) {
  const T *dout_buf = dout + threadIdx.x * 4;
  const T *input_buf = input + threadIdx.x * 4;

  float4 gamma_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 beta_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    int offset = row * hidden_dim;
    float4 dout4 = load_vector(dout_buf + offset);
    float4 input4 = load_vector(input_buf + offset);

    const float mean = (float)__ldg(&mean_[row]);
    const float var_rsqrt = (float)__ldg(&var_rsqrt_[row]);

    gamma_sum.x += dout4.x * (input4.x - mean) * var_rsqrt;
    gamma_sum.y += dout4.y * (input4.y - mean) * var_rsqrt;
    gamma_sum.z += dout4.z * (input4.z - mean) * var_rsqrt;
    gamma_sum.w += dout4.w * (input4.w - mean) * var_rsqrt;

    beta_sum.x += dout4.x;
    beta_sum.y += dout4.y;
    beta_sum.z += dout4.z;
    beta_sum.w += dout4.w;
  }

  store_vector(gamma_buf + blockIdx.x * hidden_dim + threadIdx.x * 4,
               gamma_sum);
  store_vector(beta_buf + blockIdx.x * hidden_dim + threadIdx.x * 4, beta_sum);
}

template <typename T>
__global__ void
layernorm_bw_dgamma_dbeta_reduce(const float *gamma_buf, const float *beta_buf,
                                 T *grad_gamma, T *grad_beta, int hidden_dim,
                                 int block_count) {
  __shared__ float s_gamma[32][32 + 1];
  __shared__ float s_beta[32][32 + 1];

  int warp_id = threadIdx.x >> 5;
  int warp_tid = threadIdx.x & 0x1F;

  int offset = blockIdx.x * 32 + warp_tid;
  const float *gamma = gamma_buf + offset;
  const float *beta = beta_buf + offset;

  float sum_gamma = 0.0f, sum_beta = 0.0f;
  for (int row = warp_id; row < block_count; row += 32) {
    sum_gamma += *(gamma + row * hidden_dim);
    sum_beta += *(beta + row * hidden_dim);
  }

  s_gamma[warp_tid][warp_id] = sum_gamma;
  s_beta[warp_tid][warp_id] = sum_beta;

  __syncthreads();

  float2 d_gamma_beta = warpReduceSum_2(
      make_float2(s_gamma[warp_id][warp_tid], s_beta[warp_id][warp_tid]));

  if (warp_tid == 0) {
    grad_gamma[blockIdx.x * 32 + warp_id] = (T)d_gamma_beta.x;
    grad_beta[blockIdx.x * 32 + warp_id] = (T)d_gamma_beta.y;
  }
}

template <typename T>
__global__ void layernorm_bw_dinput(const T *grad_out, const T *gamma,
                                    const T *input, const T *mean_,
                                    const T *var_rsqrt_, T *grad_in,
                                    T *grad_residual, float r_hidden_dim) {
  int offset = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

  float4 xhat4;
  float4 input4 = load_vector(input + offset);
  const float mean = (float)__ldg(&mean_[blockIdx.x]);
  const float var_rsqrt = (float)__ldg(&var_rsqrt_[blockIdx.x]);
  xhat4.x = (input4.x - mean) * var_rsqrt;
  xhat4.y = (input4.y - mean) * var_rsqrt;
  xhat4.z = (input4.z - mean) * var_rsqrt;
  xhat4.w = (input4.w - mean) * var_rsqrt;

  float4 gamma4 = load_vector(gamma + threadIdx.x * 4);
  float4 dxhat = load_vector(grad_out + offset);
  dxhat.x *= gamma4.x;
  dxhat.y *= gamma4.y;
  dxhat.z *= gamma4.z;
  dxhat.w *= gamma4.w;

  float4 dxhat_xhat;
  dxhat_xhat.x = dxhat.x * xhat4.x;
  dxhat_xhat.y = dxhat.y * xhat4.y;
  dxhat_xhat.z = dxhat.z * xhat4.z;
  dxhat_xhat.w = dxhat.w * xhat4.w;

  float2 sum2;
  sum2.x = dxhat.x + dxhat.y + dxhat.z + dxhat.w;
  sum2.y = dxhat_xhat.x + dxhat_xhat.y + dxhat_xhat.z + dxhat_xhat.w;

  sum2 = blockReduceSum_2(sum2);
  __shared__ float s_dxhat_sum, s_dxhat_xhat_sum;
  if (threadIdx.x == 0) {
    s_dxhat_sum = sum2.x;
    s_dxhat_xhat_sum = sum2.y;
  }
  __syncthreads();
  float dxhat_sum = s_dxhat_sum, dxhat_xhat_sum = s_dxhat_xhat_sum;

  float4 tmp;
  tmp.x = (xhat4.x * dxhat_xhat_sum + dxhat_sum) * r_hidden_dim;
  tmp.y = (xhat4.y * dxhat_xhat_sum + dxhat_sum) * r_hidden_dim;
  tmp.z = (xhat4.z * dxhat_xhat_sum + dxhat_sum) * r_hidden_dim;
  tmp.w = (xhat4.w * dxhat_xhat_sum + dxhat_sum) * r_hidden_dim;

  float4 result;
  result.x = (dxhat.x - tmp.x) * var_rsqrt;
  result.y = (dxhat.y - tmp.y) * var_rsqrt;
  result.z = (dxhat.z - tmp.z) * var_rsqrt;
  result.w = (dxhat.w - tmp.w) * var_rsqrt;

  if (grad_residual)
    store_vector(grad_residual + offset, result);

  store_vector(grad_in + offset, result);
}

template <typename T>
__global__ void
layernorm_bw_dinput_mini_dim(const T *grad_out, const T *gamma, const T *input,
                             const T *mean_, const T *var_rsqrt_, T *grad_in,
                             T *grad_residual, float r_hidden_dim, int rows,
                             int hidden_dim) {
  if ((blockIdx.x * blockDim.y + threadIdx.y) >= rows)
    return;
  int row = blockDim.y * blockIdx.x + threadIdx.y;
  int offset = row * hidden_dim;
  const float mean = (float)__ldg(&mean_[row]);
  const float var_rsqrt = (float)__ldg(&var_rsqrt_[row]);

  const int max_warp_per_row = 4;
  float2 sum2;
  sum2.x = 0.0f;
  sum2.y = 0.0f;
  register float s_in[max_warp_per_row];
  register float s_in2[max_warp_per_row];
  for (int idx = threadIdx.x, warp_idx = 0; idx < hidden_dim;
       idx += WARP_SIZE, warp_idx++) {
    float input_num = (float)(input + offset)[idx];
    float xhat = (input_num - mean) * var_rsqrt;
    float dxhat = (float)gamma[idx] * (float)(grad_out + offset)[idx];
    float dxhat_xhat = dxhat * xhat;

    s_in[warp_idx] = xhat;
    s_in2[warp_idx] = dxhat;
    sum2.x += dxhat;
    sum2.y += dxhat_xhat;
  }
  __syncwarp();
  sum2 = warpReduceSum_2(sum2);
  for (int idx = threadIdx.x, warp_idx = 0; idx < hidden_dim;
       idx += WARP_SIZE, warp_idx++) {
    float tmp = (s_in[warp_idx] * sum2.y + sum2.x) * r_hidden_dim;
    float result = (s_in2[warp_idx] - tmp) * var_rsqrt;

    if (grad_residual)
      (grad_residual + offset)[idx] = (T)result;
    (grad_in + offset)[idx] = (T)result;
  }
}

template <OperationType OpType>
void LayerNorm<OpType>::forward(LayerNormForwardParam param) {
  if (hidden_dim_ >= 32 * 4) {
    layernorm_fw<<<param.rows, hidden_dim_ / 4, 0, param.stream>>>(
        param.input, (const DataType_ *)(param_.gamma),
        (const DataType_ *)param_.beta, param.residual, param.mean,
        param.var_rsqrt, param.layernorm_out, param.input_add_residual,
        1.0f / hidden_dim_);
  } else {
    dim3 grid, block;
    grid.x = (param.rows + 32 - 1) / 32, block.x = 32, block.y = 32;
    // one block deal with 32 lines
    // one warp deal with one line
    layernorm_fw_mini_dim<<<grid, block, 0, param.stream>>>(
        param.input, (const DataType_ *)(param_.gamma),
        (const DataType_ *)param_.beta, param.residual, param.mean,
        param.var_rsqrt, param.layernorm_out, param.input_add_residual,
        1.0f / hidden_dim_, param.rows, hidden_dim_);
  }
}

template <OperationType OpType>
void LayerNorm<OpType>::backward(LayerNormBackwardParam param) {
  float *gamma_buf = (float *)param.buf;
  float *beta_buf = gamma_buf + block_count_ * hidden_dim_;
  dim3 grid, block;

  grid.x = block_count_, block.x = hidden_dim_ / 4;
  layernorm_bw_dgamma_dbeta_sum<<<grid, block, 0, param.stream>>>(
      param.grad_out, param.input_add_residual, param.mean, param.var_rsqrt,
      gamma_buf, beta_buf, hidden_dim_, param.rows);

  grid.x = hidden_dim_ / 32, block.x = 1024;
  layernorm_bw_dgamma_dbeta_reduce<<<grid, block, 0, param.stream>>>(
      gamma_buf, beta_buf, param.grad_gamma, param.grad_beta, hidden_dim_,
      block_count_);

  if (hidden_dim_ >= 32 * 4) {
    grid.x = param.rows, block.x = hidden_dim_ / 4;
    layernorm_bw_dinput<<<grid, block, 0, param.stream>>>(
        param.grad_out, (const DataType_ *)param_.gamma,
        param.input_add_residual, param.mean, param.var_rsqrt, param.grad_in,
        param.grad_residual, 1.0f / hidden_dim_);
  } else {
    grid.x = (param.rows + 32 - 1) / 32, block.x = 32, block.y = 32;
    layernorm_bw_dinput_mini_dim<<<grid, block, 0, param.stream>>>(
        param.grad_out, (const DataType_ *)param_.gamma,
        param.input_add_residual, param.mean, param.var_rsqrt, param.grad_in,
        param.grad_residual, 1.0f / hidden_dim_, param.rows, hidden_dim_);
  }
}

template void
LayerNorm<OperationType::FP32>::forward(LayerNormForwardParam param);
template void
LayerNorm<OperationType::HALF>::forward(LayerNormForwardParam param);

template void
LayerNorm<OperationType::FP32>::backward(LayerNormBackwardParam param);
template void
LayerNorm<OperationType::HALF>::backward(LayerNormBackwardParam param);
} // namespace fastertransformerv4