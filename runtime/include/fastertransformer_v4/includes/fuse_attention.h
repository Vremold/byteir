/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "common.h"

namespace fastertransformerv4 {
template <typename T> struct FuseAttentionForwardParam {
  const T *input_q;
  const T *input_k;
  const T *input_v;
  const T *mask;
  T *softmax_output;
  T *attention_output;
  const int batch_size;
  const int seq_len;
  const int head_num;
  const int size_per_head;
  cudaStream_t stream;
  float dropout_rate = 0.0f;
  uint8_t *dropout_mask = NULL;
  T *softmax_dropout_output = NULL;
};

template <typename T> struct FuseAttentionBackwardParam {
  const T *grad_out;
  const T *softmax_output;
  const T *input_q;
  const T *input_k;
  const T *input_v;
  T *grad_q;
  T *grad_k;
  T *grad_v;
  const int batch_size;
  const int seq_len;
  const int head_num;
  const int size_per_head;
  cudaStream_t stream;
  float dropout_rate = 0.0f;
  uint8_t *dropout_mask = NULL;
  T *softmax_dropout_output = NULL;
};

template <OperationType OpType> class FuseAttention {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

  using FuseAttentionForwardParam = struct FuseAttentionForwardParam<DataType_>;
  using FuseAttentionBackwardParam =
      struct FuseAttentionBackwardParam<DataType_>;

public:
  FuseAttention() {}

  unsigned long long cal_bufsize() { return 0; }

  void forward(FuseAttentionForwardParam param);
  void backward(FuseAttentionBackwardParam param);

  ~FuseAttention() {}
};
} // namespace fastertransformerv4
