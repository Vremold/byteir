/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v4/includes/reduce.h"
#include "fastertransformer_v4/includes/softmax.h"
#include "fastertransformer_v4/includes/utils.h"
using namespace std;

namespace fastertransformerv4 {
const int WARP_SIZE = 32;

template <typename T>
__global__ void
softmax_forward_kernel(const T *input, T *softmax_output, const int rows,
                       const int cols, const bool add_mask, const T *mask,
                       const bool batch_first, const bool apply_dropout,
                       const float dropout_rate, uint8_t *dropout_mask,
                       T *softmax_dropout_output, const int seed) {
  extern __shared__ float shmem[];
  float *s_local = shmem + (threadIdx.z * blockDim.y + threadIdx.y) * cols;

  int local_tid = threadIdx.x;
  int bs_id = blockIdx.x * blockDim.z + threadIdx.z;

  int row_id;
  if (batch_first)
    row_id = (bs_id / cols * blockDim.y + threadIdx.y) * cols + (bs_id % cols);
  else
    row_id = threadIdx.y * (gridDim.x * blockDim.z) + bs_id;

  if (row_id < rows) {
    float max_val = -1e20f, exp_sum = 0.0f;
    for (int col_id = local_tid; col_id < cols; col_id += WARP_SIZE) {
      int pos_id = row_id * cols + col_id;
      float value = (float)__ldg(&input[pos_id]);
      if (add_mask)
        value +=
            (1.0f - (float)__ldg(&mask[bs_id * cols + col_id])) * -10000.0f;
      s_local[col_id] = value;
      max_val = value > max_val ? value : max_val;
    }
    max_val = warpReduceMax<float>(max_val);

    for (int col_id = local_tid; col_id < cols; col_id += WARP_SIZE) {
      float exp_val = __expf(s_local[col_id] - max_val);
      s_local[col_id] = exp_val;
      exp_sum += exp_val;
    }
    exp_sum = warpReduceSum<float>(exp_sum) + 1e-6f;
    exp_sum = __fdividef(1.0f, exp_sum);
    for (int col_id = local_tid; col_id < cols; col_id += WARP_SIZE) {
      int pos_id = row_id * cols + col_id;
      T out_val = (T)(s_local[col_id] * exp_sum);
      softmax_output[pos_id] = out_val;
      if (apply_dropout) {
        out_val = dropout_fw(out_val, dropout_rate, seed, pos_id, dropout_mask);
        softmax_dropout_output[pos_id] = out_val;
      }
    }
  }
}

template <>
__global__ void softmax_forward_kernel(
    const __half *input, __half *softmax_output, const int rows, const int cols,
    const bool add_mask, const __half *mask, const bool batch_first,
    const bool apply_dropout, const float dropout_rate, uint8_t *dropout_mask,
    __half *softmax_dropout_output, const int seed) {
  extern __shared__ float shmem[];
  float *s_local = shmem + (threadIdx.z * blockDim.y + threadIdx.y) * cols;

  int half_cols = cols / 2;
  const half2 *input_ptr = (const half2 *)(input);
  const half2 *mask_ptr = (const half2 *)(mask);
  half2 *softmax_output_ptr = (half2 *)(softmax_output);
  half2 *softmax_dropout_output_ptr = (half2 *)(softmax_dropout_output);

  int local_tid = threadIdx.x;
  int bs_id = blockIdx.x * blockDim.z + threadIdx.z;
  int row_id;
  if (batch_first)
    row_id = (bs_id / cols * blockDim.y + threadIdx.y) * cols + (bs_id % cols);
  else
    row_id = threadIdx.y * (gridDim.x * blockDim.z) + bs_id;

  if (row_id < rows) {
    float max_val = -1e20f, exp_sum = 0.0f;
    for (int col_id = local_tid; col_id < half_cols; col_id += WARP_SIZE) {
      int pos_id = row_id * half_cols + col_id;
      float2 value = __half22float2(__ldg(&input_ptr[pos_id]));
      if (add_mask) {
        float2 mask_val =
            __half22float2(__ldg(&mask_ptr[bs_id * half_cols + col_id]));
        value.x += (1.0f - mask_val.x) * -10000.0f;
        value.y += (1.0f - mask_val.y) * -10000.0f;
      }

      s_local[col_id * 2] = value.x;
      s_local[col_id * 2 + 1] = value.y;
      max_val = value.x > max_val ? value.x : max_val;
      max_val = value.y > max_val ? value.y : max_val;
    }
    max_val = warpReduceMax<float>(max_val);

    for (int col_id = local_tid; col_id < cols; col_id += WARP_SIZE) {
      float exp_val = __expf(s_local[col_id] - max_val);
      s_local[col_id] = exp_val;
      exp_sum += exp_val;
    }
    exp_sum = warpReduceSum<float>(exp_sum) + 1e-6f;
    exp_sum = __fdividef(1.0f, exp_sum);
    for (int col_id = local_tid; col_id < half_cols; col_id += WARP_SIZE) {
      int pos_id = row_id * half_cols + col_id;

      half2 val;
      val.x = s_local[col_id * 2] * exp_sum;
      val.y = s_local[col_id * 2 + 1] * exp_sum;
      softmax_output_ptr[pos_id] = val;
      if (apply_dropout) {
        val =
            dropout_fw(val, dropout_rate, seed, pos_id, (uchar2 *)dropout_mask);
        softmax_dropout_output_ptr[pos_id] = val;
      }
    }
  }
}

template <typename T>
__global__ void
softmax_backward_kernel(const T *grad_out, const T *softmax_out, T *grad_in,
                        const int rows, const int cols,
                        const bool apply_dropout, const float scale,
                        const uint8_t *dropout_mask) {
  int total_warps = gridDim.x * blockDim.x / WARP_SIZE;
  int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
  int local_tid = threadIdx.x % WARP_SIZE;

  extern __shared__ unsigned char smem[];
  T *s_out = (reinterpret_cast<T *>(smem)) + threadIdx.x / WARP_SIZE * cols;
  T *s_grad = (reinterpret_cast<T *>(smem)) + threadIdx.x / WARP_SIZE * cols +
              blockDim.x / WARP_SIZE * cols;

  for (int row_id = warp_id; row_id < rows; row_id += total_warps) {
    T sum = (T)(0.0f);
    for (int col_id = local_tid; col_id < cols; col_id += WARP_SIZE) {
      int pos_id = row_id * cols + col_id;
      T out = __ldg(&softmax_out[pos_id]);
      T grad = __ldg(&grad_out[pos_id]);
      if (apply_dropout)
        grad = dropout_mask[pos_id] ? grad * scale : 0.0f;

      sum += out * grad;
      s_out[col_id] = out;
      s_grad[col_id] = grad;
    }
    sum = warpReduceSum<float>(sum);

    for (int col_id = local_tid; col_id < cols; col_id += WARP_SIZE)
      grad_in[row_id * cols + col_id] = s_out[col_id] * (s_grad[col_id] - sum);
  }
}

template <>
__global__ void
softmax_backward_kernel(const __half *grad_out, const __half *softmax_out,
                        __half *grad_in, const int rows, const int cols,
                        const bool apply_dropout, const float scale,
                        const uint8_t *dropout_mask) {
  int total_warps = gridDim.x * blockDim.x / WARP_SIZE;
  int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
  int local_tid = threadIdx.x % WARP_SIZE;

  extern __shared__ unsigned char smem[];

  int half2_cols = cols / 2;
  half2 *s_out =
      (reinterpret_cast<half2 *>(smem)) + threadIdx.x / WARP_SIZE * half2_cols;
  half2 *s_grad = (reinterpret_cast<half2 *>(smem)) +
                  threadIdx.x / WARP_SIZE * half2_cols +
                  blockDim.x / WARP_SIZE * half2_cols;

  const half2 *softmax_out_ptr = (const half2 *)(softmax_out);
  const half2 *grad_out_ptr = (const half2 *)(grad_out);
  half2 *grad_in_ptr = (half2 *)(grad_in);

  half2 zero_half(0.0f, 0.0f);
  for (int row_id = warp_id; row_id < rows; row_id += total_warps) {
    half2 sum = zero_half;
    for (int col_id = local_tid; col_id < half2_cols; col_id += WARP_SIZE) {
      int pos_id = row_id * half2_cols + col_id;
      half2 out = __ldg(&softmax_out_ptr[pos_id]);
      half2 grad = __ldg(&grad_out_ptr[pos_id]);
      if (apply_dropout)
        grad = dropout_bw(grad, scale, pos_id, (const uchar2 *)dropout_mask);

      sum = __hfma2(out, grad, sum);
      s_out[col_id] = out;
      s_grad[col_id] = grad;
    }
    __half reduce_sum = warpReduceSum<float>((float)(sum.x + sum.y));
    sum.x = reduce_sum;
    sum.y = reduce_sum;
    for (int col_id = local_tid; col_id < half2_cols; col_id += WARP_SIZE)
      grad_in_ptr[row_id * half2_cols + col_id] =
          __hmul2(s_out[col_id], __hsub2(s_grad[col_id], sum));
  }
}

template <OperationType OpType>
void Softmax<OpType>::forward(SoftmaxForwardParam param) {
  const int seed = generate_random_seed();

  dim3 grid, block;
  block.x = WARP_SIZE;
  block.y = param.head_num;                    // <= 12
  block.z = max(384 / (block.x * block.y), 1); // thread count <= 384 (32 * 12)
  grid.x = ((param.rows / param.head_num) + block.z - 1) / block.z;
  size_t shared_mem_size =
      param.cols * (block.x * block.y * block.z / WARP_SIZE) *
      sizeof(float); // shared memory <=48KB, support seq_len <= 1024

  softmax_forward_kernel<<<grid, block, shared_mem_size, param.stream>>>(
      param.input, param.softmax_output, param.rows, param.cols, param.add_mask,
      param.mask, param.batch_first, param.apply_dropout, param.dropout_rate,
      param.dropout_mask, param.softmax_dropout_output, seed);
}

template <OperationType OpType>
void Softmax<OpType>::backward(SoftmaxBackwardParam param) {
  float scale = 1.0f / (1.0f - param.dropout_rate);

  dim3 grid(320), block(128);
  size_t shared_mem_size = param.cols * (block.x / WARP_SIZE) *
                           sizeof(DataType_) *
                           2; // shared memory <=48KB, support seq_len <= 1024

  softmax_backward_kernel<<<grid, block, shared_mem_size, param.stream>>>(
      param.grad_out, param.out, param.grad_in, param.rows, param.cols,
      param.apply_dropout, scale, param.dropout_mask);
}

template void Softmax<OperationType::FP32>::forward(SoftmaxForwardParam param);
template void Softmax<OperationType::HALF>::forward(SoftmaxForwardParam param);

template void
Softmax<OperationType::FP32>::backward(SoftmaxBackwardParam param);
template void
Softmax<OperationType::HALF>::backward(SoftmaxBackwardParam param);
} // namespace fastertransformerv4