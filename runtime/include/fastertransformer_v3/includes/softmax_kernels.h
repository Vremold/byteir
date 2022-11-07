/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "fastertransformer_v3/includes/common.h"
#include <cuda_fp16.h>

namespace fastertransformerv3 {

template <typename T>
__global__ void softmax_kernel(T *qk_buf_, const T *attr_mask,
                               const int batch_size, const int head_num,
                               const int seq_len, const T scaler) {
  int batch_id = blockIdx.x / head_num;
  int qk_offset = blockIdx.x * seq_len * seq_len;
  int mask_offset = batch_id * seq_len * seq_len;

  __shared__ float s_sum, s_max;

  for (int i = 0; i < seq_len; ++i) {
    float qk =
        threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    float mask_val = threadIdx.x < seq_len
                         ? (float)attr_mask[threadIdx.x + mask_offset]
                         : 0.0f;

    mask_val = (1.0f - mask_val) * -10000.0f;

    float tmp =
        threadIdx.x < seq_len ? (float)(qk * (float)scaler + mask_val) : -1e20f;

    float max_val = blockReduceMax<float>(tmp);

    if (threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

    float sum_val = blockReduceSum<float>(qk);

    if (threadIdx.x == 0)
      s_sum = sum_val + 1e-6f;
    __syncthreads();

    if (threadIdx.x < seq_len)
      qk_buf_[threadIdx.x + qk_offset] = (T)(qk / s_sum);

    qk_offset += seq_len;
    mask_offset += seq_len;
  }
}

template <typename T>
__global__ void softmax_kernel_v2(T *qk_buf_, const T *attr_mask,
                                  const int batch_size, const int head_num,
                                  const int seq_len, const float scaler) {
  int batch_id = blockIdx.x / head_num / seq_len;
  int seq_id = blockIdx.x % seq_len;
  int qk_offset = blockIdx.x * seq_len;
  int mask_offset = batch_id * seq_len * seq_len + seq_id * seq_len;

  __shared__ float s_sum, s_max;

  float qk =
      threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
  float mask_val = threadIdx.x < seq_len
                       ? (float)attr_mask[threadIdx.x + mask_offset]
                       : 0.0f;

  mask_val = (1.0f - mask_val) * -10000.0f;

  float tmp =
      threadIdx.x < seq_len ? (float)(qk * (float)scaler + mask_val) : -1e20f;
  float max_val = blockReduceMax<float>(tmp);
  if (threadIdx.x == 0)
    s_max = max_val;
  __syncthreads();

  float qk_tmp = threadIdx.x < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
  float sum_val = blockReduceSum<float>(qk_tmp);

  if (threadIdx.x == 0)
    s_sum = sum_val + 1e-6f;
  __syncthreads();

  if (threadIdx.x < seq_len)
    qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

template <typename T>
__global__ void softmax_kernel_v3(T *qk_buf_, const T *attr_mask,
                                  const int batch_size, const int head_num,
                                  const int seq_len, const float scaler) {
  extern __shared__ float shmem[];
  int batch_id = blockIdx.x / head_num;
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  int w_num = blockDim.x >> 5;
  int qk_offset = blockIdx.x * seq_len * seq_len + wid * seq_len;
  int mask_offset = batch_id * seq_len * seq_len + wid * seq_len;
  float *s_row_qk = shmem + wid * seq_len;
  const int offset = w_num * seq_len;

  for (int row_id = wid; row_id < seq_len; row_id += w_num) {
    float max_v = -1e20f, exp_sum = 0.0f;
    for (int col_id = lane; col_id < seq_len; col_id += warpSize) {
      float qk = (float)qk_buf_[qk_offset + col_id];
      float mask_val = (float)attr_mask[mask_offset + col_id];
      mask_val = (1.0f - mask_val) * -10000.0f;
      float tmp = qk * scaler + mask_val;
      s_row_qk[col_id] = tmp;
      max_v = tmp > max_v ? tmp : max_v;
    }
    max_v = warpReduceMax<float>(max_v);
    for (int col_id = lane; col_id < seq_len; col_id += warpSize) {
      float qk = __expf(s_row_qk[col_id] - max_v);
      s_row_qk[col_id] = qk;
      exp_sum += qk;
    }
    exp_sum = warpReduceSum<float>(exp_sum);
    exp_sum = 1.0f / (exp_sum + 1e-6f);
    for (int col_id = lane; col_id < seq_len; col_id += warpSize)
      qk_buf_[qk_offset + col_id] = (T)(s_row_qk[col_id] * exp_sum);

    qk_offset += offset;
    mask_offset += offset;
  }
}

template <typename T>
__global__ void softmax_kernel_v3_half2(half2 *qk_buf_, const half2 *attr_mask,
                                        const int batch_size,
                                        const int head_num, const int seq_len,
                                        const float scaler) {
  extern __shared__ float shmem[];
  int batch_id = blockIdx.x / head_num;
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  int w_num = blockDim.x >> 5;
  int half2_seq_len = seq_len / 2;
  int qk_offset = blockIdx.x * seq_len * half2_seq_len + wid * half2_seq_len;
  int mask_offset = batch_id * seq_len * half2_seq_len + wid * half2_seq_len;
  const int offset = w_num * half2_seq_len;
  float *s_qk_buf = (float *)shmem + wid * seq_len;

  for (int row_id = wid; row_id < seq_len; row_id += w_num) {
    float max_val = -1e20f, exp_sum = 0.0f;
    for (int col_id = lane; col_id < half2_seq_len; col_id += warpSize) {
      half2 qk = qk_buf_[qk_offset + col_id];
      half2 mask_val = attr_mask[mask_offset + col_id];
      float qk_x = (float)qk.x, qk_y = (float)qk.y;
      float mask_val_x = (float)mask_val.x, mask_val_y = (float)mask_val.y;
      mask_val_x = (1.0f - mask_val_x) * -10000.0f;
      mask_val_y = (1.0f - mask_val_y) * -10000.0f;
      float tmp_x = scaler * qk_x + mask_val_x;
      float tmp_y = scaler * qk_y + mask_val_y;
      s_qk_buf[col_id * 2] = tmp_x;
      s_qk_buf[col_id * 2 + 1] = tmp_y;
      max_val = fmax(max_val, fmax(tmp_x, tmp_y));
    }
    max_val = warpReduceMax<float>(max_val);
    for (int col_id = lane; col_id < seq_len; col_id += warpSize) {
      float qk = __expf(s_qk_buf[col_id] - max_val);
      s_qk_buf[col_id] = qk;
      exp_sum += qk;
    }
    exp_sum = warpReduceSum<float>(exp_sum);
    exp_sum = 1.0f / (exp_sum + 1e-6f);
    for (int col_id = lane; col_id < half2_seq_len; col_id += warpSize)
      qk_buf_[qk_offset + col_id] =
          __halves2half2((half)(s_qk_buf[col_id * 2] * exp_sum),
                         (half)(s_qk_buf[col_id * 2 + 1] * exp_sum));

    qk_offset += offset;
    mask_offset += offset;
  }
}

template <typename T>
__global__ void softmax_kernel_v3_et(T *qk_buf_, const T *attr_mask,
                                     const int batch_size, const int head_num,
                                     const int seq_len, const float scaler,
                                     int *batch_idx, int *word_idx) {
  extern __shared__ float shmem[];
  int batch_id = blockIdx.x / head_num;
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  int w_num = blockDim.x >> 5;
  int qk_offset = blockIdx.x * seq_len * seq_len + wid * seq_len;
  int mask_offset = batch_id * seq_len * seq_len + wid * seq_len;
  float *s_row_qk = shmem + wid * seq_len;
  const int offset = w_num * seq_len;

  const int batch_seq_len =
      __ldg(&batch_idx[batch_id + 1]) - __ldg(&batch_idx[batch_id]);

  for (int row_id = wid; row_id < batch_seq_len; row_id += w_num) {
    float max_v = -1e20f, exp_sum = 0.0f;
    for (int col_id = lane; col_id < batch_seq_len; col_id += warpSize) {
      float qk = (float)qk_buf_[qk_offset + col_id];
      float mask_val = (float)attr_mask[mask_offset + col_id];
      mask_val = (1.0f - mask_val) * -10000.0f;
      float tmp = qk * scaler + mask_val;
      s_row_qk[col_id] = tmp;
      max_v = tmp > max_v ? tmp : max_v;
    }
    max_v = warpReduceMax<float>(max_v);
    for (int col_id = lane; col_id < batch_seq_len; col_id += warpSize) {
      float qk = __expf(s_row_qk[col_id] - max_v);
      s_row_qk[col_id] = qk;
      exp_sum += qk;
    }
    exp_sum = warpReduceSum<float>(exp_sum);
    exp_sum = 1.0f / (exp_sum + 1e-6f);
    for (int col_id = lane; col_id < batch_seq_len; col_id += warpSize)
      qk_buf_[qk_offset + col_id] = (T)(s_row_qk[col_id] * exp_sum);

    qk_offset += offset;
    mask_offset += offset;
  }
}

template <typename T>
__global__ void
softmax_kernel_v3_half2_et(half2 *qk_buf_, const half2 *attr_mask,
                           const int batch_size, const int head_num,
                           const int seq_len, const float scaler,
                           int *batch_idx, int *word_idx) {
  extern __shared__ float shmem[];
  int batch_id = blockIdx.x / head_num;
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  int w_num = blockDim.x >> 5;
  int half2_seq_len = seq_len / 2;
  int qk_offset = blockIdx.x * seq_len * half2_seq_len + wid * half2_seq_len;
  int mask_offset = batch_id * seq_len * half2_seq_len + wid * half2_seq_len;
  const int offset = w_num * half2_seq_len;
  float *s_qk_buf = (float *)shmem + wid * seq_len;

  const int batch_seq_len =
      __ldg(&batch_idx[batch_id + 1]) - __ldg(&batch_idx[batch_id]);
  const int half2_batch_seq_len = (batch_seq_len + 1) >> 1;

  for (int row_id = wid; row_id < batch_seq_len; row_id += w_num) {
    float max_val = -1e20f, exp_sum = 0.0f;
    for (int col_id = lane; col_id < half2_batch_seq_len; col_id += warpSize) {
      half2 qk = qk_buf_[qk_offset + col_id];
      half2 mask_val = attr_mask[mask_offset + col_id];
      float qk_x = (float)qk.x, qk_y = (float)qk.y;
      float mask_val_x = (float)mask_val.x, mask_val_y = (float)mask_val.y;
      mask_val_x = (1.0f - mask_val_x) * -10000.0f;
      mask_val_y = (1.0f - mask_val_y) * -10000.0f;
      float tmp_x = scaler * qk_x + mask_val_x;
      float tmp_y = scaler * qk_y + mask_val_y;
      s_qk_buf[col_id * 2] = tmp_x;
      s_qk_buf[col_id * 2 + 1] = tmp_y;
      max_val = fmax(max_val, fmax(tmp_x, tmp_y));
    }
    max_val = warpReduceMax<float>(max_val);
    for (int col_id = lane; col_id < batch_seq_len; col_id += warpSize) {
      float qk = __expf(s_qk_buf[col_id] - max_val);
      s_qk_buf[col_id] = qk;
      exp_sum += qk;
    }
    exp_sum = warpReduceSum<float>(exp_sum);
    exp_sum = 1.0f / (exp_sum + 1e-6f);
    for (int col_id = lane; col_id < half2_batch_seq_len; col_id += warpSize)
      qk_buf_[qk_offset + col_id] =
          __halves2half2((half)(s_qk_buf[col_id * 2] * exp_sum),
                         (half)(s_qk_buf[col_id * 2 + 1] * exp_sum));

    qk_offset += offset;
    mask_offset += offset;
  }
}

// Todo:
// 1. (done) softmax for et_nofused using batch_idx[], word_idx[] to reduce
// compute
// 2. (done) softmax using half2
// 3. (done) softmax for longer seq_len (use warpReduce)
// 4. (wmma) softmax for seq_len <= 32  (use warpReduce)

template <OperationType OpType_, typename T>
void softmax_kernelLauncher(T *qk_buf_, const T *atten_mask,
                            const int batch_size, const int seq_len,
                            const int head_num_, const int size_per_head_,
                            cudaStream_t stream, const bool no_scale = false) {
  T scaler;

  if (no_scale)
    scaler = (T)1.0f;
  else
    scaler = 1.0f / sqrtf(size_per_head_ * 1.0f);

  dim3 grid, block;

  if (batch_size * head_num_ * seq_len <= 120 * 16) {
    grid.x = batch_size * head_num_ * seq_len;
    block.x = seq_len > 1024 ? 1024 : ((seq_len + 31) / 32) * 32;
    softmax_kernel_v2<<<grid, block, 0, stream>>>(
        qk_buf_, atten_mask, batch_size, head_num_, seq_len, scaler);
  } else {
    const int warp_num = 8;
    grid.x = batch_size * head_num_;
    block.x = warp_num * 32;
    int shmem_size = warp_num * seq_len * sizeof(float);

    if ((seq_len & 0x1) == 0 && OpType_ == OperationType::HALF)
      softmax_kernel_v3_half2<half2><<<grid, block, shmem_size, stream>>>(
          (half2 *)qk_buf_, (half2 *)atten_mask, batch_size, head_num_, seq_len,
          scaler);
    else
      softmax_kernel_v3<T><<<grid, block, shmem_size, stream>>>(
          qk_buf_, atten_mask, batch_size, head_num_, seq_len, scaler);
  }
}

template <OperationType OpType_, typename T>
void softmax_et_kernelLauncher(T *qk_buf_, const T *atten_mask,
                               const int batch_size, const int seq_len,
                               const int head_num_, const int size_per_head_,
                               cudaStream_t stream, int *batch_idx,
                               int *word_idx) {
  T scaler = 1 / sqrtf(size_per_head_ * 1.0f);

  dim3 grid, block;

  if (batch_size * head_num_ * seq_len <= 120 * 16) {
    grid.x = batch_size * head_num_ * seq_len;
    block.x = seq_len > 1024 ? 1024 : ((seq_len + 31) / 32) * 32;
    softmax_kernel_v2<<<grid, block, 0, stream>>>(
        qk_buf_, atten_mask, batch_size, head_num_, seq_len, scaler);
  } else {
    const int warp_num = 8;
    grid.x = batch_size * head_num_;
    block.x = warp_num * 32;
    int shmem_size = warp_num * seq_len * sizeof(float);

    if ((seq_len & 0x1) == 0 && OpType_ == OperationType::HALF)
      softmax_kernel_v3_half2_et<half2><<<grid, block, shmem_size, stream>>>(
          (half2 *)qk_buf_, (half2 *)atten_mask, batch_size, head_num_, seq_len,
          scaler, batch_idx, word_idx);
    else
      softmax_kernel_v3_et<T><<<grid, block, shmem_size, stream>>>(
          qk_buf_, atten_mask, batch_size, head_num_, seq_len, scaler,
          batch_idx, word_idx);
  }
}

} // namespace fastertransformerv3
