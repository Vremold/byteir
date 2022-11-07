/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v4/includes/reduce.h"
#include "fastertransformer_v4/includes/search_ctr_mha.h"
#include "fastertransformer_v4/includes/utils.h"
using namespace std;

namespace fastertransformerv4 {
template <typename T, const int MAX_SEQ_LEN, const int SIZE_PER_HEAD>
__global__ void fuse_attention_forward_kernel(
    const T *input_q, const T *input_k, const T *input_v, const T *mask,
    T *softmax_out, T *attention_out, const int batch_size, const int seq_len,
    const int head_num, const float scaler) {
  __shared__ T s_buf1[MAX_SEQ_LEN][SIZE_PER_HEAD + 1]; // for grad_out
  __shared__ T s_buf2[MAX_SEQ_LEN][SIZE_PER_HEAD + 1]; // loading q/k/v tensor
  __shared__ T s_buf3[MAX_SEQ_LEN][MAX_SEQ_LEN + 1];   // softmax_out

  int batch_id = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  // s_buf1 loading Q/K
  for (int tid = threadIdx.x; tid < seq_len * SIZE_PER_HEAD;
       tid += blockDim.x) {
    int seq_id = tid / SIZE_PER_HEAD;
    int dim_id = tid % SIZE_PER_HEAD;

    int id = batch_id * seq_len * head_num * SIZE_PER_HEAD +
             seq_id * head_num * SIZE_PER_HEAD + head_id * SIZE_PER_HEAD +
             dim_id;

    s_buf1[seq_id][dim_id] = __ldg(&input_q[id]);
    s_buf2[seq_id][dim_id] = __ldg(&input_k[id]);
  }
  __syncthreads();

  for (int tid = threadIdx.x; tid < seq_len * seq_len; tid += blockDim.x) {
    int seq_id1 = tid / seq_len;
    int seq_id2 = tid % seq_len;
    T qk_val = 0.0f;
    for (int dim_id = 0; dim_id < SIZE_PER_HEAD; dim_id++)
      qk_val += s_buf1[seq_id1][dim_id] * s_buf2[seq_id2][dim_id];
    qk_val *= scaler;
    qk_val += ((T)1.0f - mask[batch_id * seq_len * seq_len + tid]) * (T)-1e4f;
    s_buf3[seq_id1][seq_id2] = qk_val;
  }
  __syncthreads();
  // s_buf3 qk [seq_len][seq_len]
  // softmax
  const int WARP_SIZE = 32;
  int total_warps = blockDim.x / WARP_SIZE;
  int local_tid = threadIdx.x % WARP_SIZE;
  for (int seq_id = threadIdx.x / WARP_SIZE; seq_id < seq_len;
       seq_id += total_warps) {
    float max_val = -1e20f, exp_sum = 0.0f;
    for (int tid = local_tid; tid < seq_len; tid += WARP_SIZE)
      max_val = (float)s_buf3[seq_id][tid] > max_val
                    ? (float)s_buf3[seq_id][tid]
                    : max_val;
    max_val = warpReduceMax<float>(max_val);
    for (int tid = local_tid; tid < seq_len; tid += WARP_SIZE) {
      float exp_val = __expf((float)s_buf3[seq_id][tid] - max_val);
      s_buf3[seq_id][tid] = exp_val;
      exp_sum += exp_val;
    }

    exp_sum = warpReduceSum<float>(exp_sum) + 1e-6f;
    exp_sum = __fdividef(1.0f, exp_sum);
    for (int tid = local_tid; tid < seq_len; tid += WARP_SIZE) {
      T out_val = (T)((float)s_buf3[seq_id][tid] * exp_sum);
      s_buf3[seq_id][tid] = out_val;
      int id = blockIdx.x * seq_len * seq_len + seq_id * seq_len + tid;
      softmax_out[id] = out_val;
    }
  }
  // s_buf1 loading Q/K/V
  for (int tid = threadIdx.x; tid < seq_len * SIZE_PER_HEAD;
       tid += blockDim.x) {
    int seq_id = tid / SIZE_PER_HEAD;
    int dim_id = tid % SIZE_PER_HEAD;

    int id = batch_id * seq_len * head_num * SIZE_PER_HEAD +
             seq_id * head_num * SIZE_PER_HEAD + head_id * SIZE_PER_HEAD +
             dim_id;
    s_buf1[seq_id][dim_id] = __ldg(&input_v[id]);
  }
  __syncthreads();

  // softmax * V
  // s_buf3 [seq_len][seq_len]
  // s_buf1 [seq_len][size_per_head]
  for (int tid = threadIdx.x; tid < seq_len * SIZE_PER_HEAD;
       tid += blockDim.x) {
    int seq_id = tid / SIZE_PER_HEAD;
    int dim_id = tid % SIZE_PER_HEAD;
    T val = 0.0f;

    for (int id = 0; id < seq_len; id++)
      val += s_buf3[seq_id][id] * s_buf1[id][dim_id];

    int id = batch_id * seq_len * head_num * SIZE_PER_HEAD +
             seq_id * head_num * SIZE_PER_HEAD + head_id * SIZE_PER_HEAD +
             dim_id;
    attention_out[id] = val;
  }
}

// grad_out: [batch_size, seq_len, head_num, size_per_head]
// softmax_out: [batch_size, head_num, seq_len, seq_len]
// input_q/k/v: [batch_size, seq_len, head_num, size_per_head]

template <typename T, const int MAX_SEQ_LEN, const int SIZE_PER_HEAD>
__global__ void fuse_attention_backward_kernel(
    const T *grad_out, const T *softmax_output, const T *input_q,
    const T *input_k, const T *input_v, T *grad_q, T *grad_k, T *grad_v,
    const int batch_size, const int seq_len, const int head_num,
    const float scaler) {
  __shared__ T s_buf1[MAX_SEQ_LEN][SIZE_PER_HEAD + 1]; // for grad_out
  __shared__ T s_buf2[MAX_SEQ_LEN][MAX_SEQ_LEN + 1];   // softmax_out
  __shared__ T s_buf3[MAX_SEQ_LEN][MAX_SEQ_LEN + 1];   // softmax_out_grad
  __shared__ T s_buf4[MAX_SEQ_LEN][SIZE_PER_HEAD + 1]; // loading q/k/v tensor

  int batch_id = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  // s_buf1 grad out
  for (int tid = threadIdx.x; tid < seq_len * SIZE_PER_HEAD;
       tid += blockDim.x) {
    int seq_id = tid / SIZE_PER_HEAD;
    int dim_id = tid % SIZE_PER_HEAD;

    int grad_id = batch_id * seq_len * head_num * SIZE_PER_HEAD +
                  seq_id * head_num * SIZE_PER_HEAD + head_id * SIZE_PER_HEAD +
                  dim_id;

    s_buf1[seq_id][dim_id] = __ldg(&grad_out[grad_id]);
    s_buf4[seq_id][dim_id] = __ldg(&input_v[grad_id]);
  }

  // s_buf2 softmax_output
  for (int tid = threadIdx.x; tid < seq_len * seq_len; tid += blockDim.x) {
    int seq_id1 = tid / seq_len;
    int seq_id2 = tid % seq_len;
    s_buf2[seq_id1][seq_id2] =
        __ldg(&softmax_output[blockIdx.x * seq_len * seq_len + tid]);
  }
  __syncthreads();

  // test_v_grad = tf.matmul(tf.transpose(logits4, perm=[0, 1, 3, 2]),
  // logits5_grad[0]) compute v_grad [batch_size, head_num, seq_len,
  // size_per_head] [seq_len, seq_len] x [seq_len, size_per_head]
  for (int tid = threadIdx.x; tid < seq_len * SIZE_PER_HEAD;
       tid += blockDim.x) {
    int seq_id = tid / SIZE_PER_HEAD;
    int dim_id = tid % SIZE_PER_HEAD;

    T v_grad_val = (T)0.0f;
    for (int id = 0; id < seq_len; id++) {
      v_grad_val += s_buf2[id][seq_id] * s_buf1[id][dim_id];
    }
    grad_v[batch_id * head_num * seq_len * SIZE_PER_HEAD +
           seq_id * head_num * SIZE_PER_HEAD + head_id * SIZE_PER_HEAD +
           dim_id] = v_grad_val;
  }

  // test_logits4_grad = tf.matmul(logits5_grad[0], tf.transpose(v, perm=[0, 1,
  // 3, 2])) s_buf1 logits5_grad [seq_len][dim] s_buf4 v [seq_len][dim] s_buf3
  // softmax_out_grad [seq_len][seq_len]
  for (int tid = threadIdx.x; tid < seq_len * seq_len; tid += blockDim.x) {
    int seq_id1 = tid / seq_len;
    int seq_id2 = tid % seq_len;

    T grad_val = (T)0.0f;
    for (int id = 0; id < SIZE_PER_HEAD; id++) {
      grad_val += s_buf1[seq_id1][id] * s_buf4[seq_id2][id];
    }
    s_buf3[seq_id1][seq_id2] = grad_val;
  }
  __syncthreads();

  // softmax input grad
  // s_buf3 softmax_output_grad [seq_len][seq_len]
  // s_buf2 softmax_out [seq_len][seq_len]
  // put in s_buf1
  const int WARP_SIZE = 32;
  int total_warps = blockDim.x / WARP_SIZE;
  int local_tid = threadIdx.x % WARP_SIZE;
  for (int seq_id = threadIdx.x / WARP_SIZE; seq_id < seq_len;
       seq_id += total_warps) {
    T sum_val = 0.0f;
    for (int tid = local_tid; tid < seq_len; tid += WARP_SIZE) {
      sum_val += s_buf2[seq_id][tid] * s_buf3[seq_id][tid];
    }
    sum_val = warpReduceSum<float>(sum_val);
    for (int tid = local_tid; tid < seq_len; tid += WARP_SIZE)
      s_buf2[seq_id][tid] =
          s_buf2[seq_id][tid] * (s_buf3[seq_id][tid] - sum_val) * (T)scaler;
  }
  // s_buf1 grad out
  for (int tid = threadIdx.x; tid < seq_len * SIZE_PER_HEAD;
       tid += blockDim.x) {
    int seq_id = tid / SIZE_PER_HEAD;
    int dim_id = tid % SIZE_PER_HEAD;

    int id = batch_id * seq_len * head_num * SIZE_PER_HEAD +
             seq_id * head_num * SIZE_PER_HEAD + head_id * SIZE_PER_HEAD +
             dim_id;

    s_buf1[seq_id][dim_id] = __ldg(&input_q[id]);
    s_buf4[seq_id][dim_id] = __ldg(&input_k[id]);
  }
  __syncthreads();

  // s_buf2 grad_in [seq_len, seq_len]
  // s_buf1 input_q [seq_len, dim]
  // s_buf4 input_k [seq_len, dim]
  for (int tid = threadIdx.x; tid < seq_len * SIZE_PER_HEAD;
       tid += blockDim.x) {
    int seq_id = tid / SIZE_PER_HEAD;
    int dim_id = tid % SIZE_PER_HEAD;

    T grad_q_val = (T)0.0f;
    T grad_k_val = (T)0.0f;
    for (int id = 0; id < seq_len; id++) {
      grad_q_val += s_buf2[seq_id][id] * s_buf4[id][dim_id];
      grad_k_val += s_buf2[id][seq_id] * s_buf1[id][dim_id];
    }

    int grad_id = batch_id * head_num * seq_len * SIZE_PER_HEAD +
                  seq_id * head_num * SIZE_PER_HEAD + head_id * SIZE_PER_HEAD +
                  dim_id;

    grad_k[grad_id] = grad_k_val;
    grad_q[grad_id] = grad_q_val;
  }
}

template <OperationType OpType>
void FuseAttentionCTR<OpType>::forward(FuseAttentionCTRForwardParam param) {
  const int batch_size = param.batch_size;
  const int seq_len = param.seq_len;
  const int head_num = param.head_num;
  // const int size_per_head = param.size_per_head;

  dim3 grid(batch_size * head_num);
  dim3 block(1024);

  fuse_attention_forward_kernel<DataType_, 64, 16>
      <<<grid, block, 0, param.stream>>>(
          param.input_q, param.input_k, param.input_v, param.mask,
          param.softmax_output, param.attention_output, batch_size, seq_len,
          head_num, param.scaler);
}

template <OperationType OpType>
void FuseAttentionCTR<OpType>::backward(FuseAttentionCTRBackwardParam param) {
  const int batch_size = param.batch_size;
  const int seq_len = param.seq_len;
  const int head_num = param.head_num;
  // const int size_per_head = param.size_per_head;

  dim3 grid(batch_size * head_num);
  dim3 block(1024);

  fuse_attention_backward_kernel<DataType_, 64, 16>
      <<<grid, block, 0, param.stream>>>(
          param.grad_out, param.softmax_output, param.input_q, param.input_k,
          param.input_v, param.grad_q, param.grad_k, param.grad_v, batch_size,
          seq_len, head_num, param.scaler);
}

template void FuseAttentionCTR<OperationType::FP32>::forward(
    FuseAttentionCTRForwardParam param);
template void FuseAttentionCTR<OperationType::HALF>::forward(
    FuseAttentionCTRForwardParam param);

template void FuseAttentionCTR<OperationType::FP32>::backward(
    FuseAttentionCTRBackwardParam param);
template void FuseAttentionCTR<OperationType::HALF>::backward(
    FuseAttentionCTRBackwardParam param);
} // namespace fastertransformerv4