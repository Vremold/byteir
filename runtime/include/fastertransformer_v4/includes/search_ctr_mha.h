/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "common.h"

namespace fastertransformerv4 {
template <typename T> struct FuseAttentionCTRForwardParam {
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
  const float dropout_rate;
  uint8_t *dropout_mask;
  T *softmax_dropout_output;
  const float scaler;
};

template <typename T> struct FuseAttentionCTRBackwardParam {
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
  const float dropout_rate;
  const uint8_t *dropout_mask;
  const T *softmax_dropout_output;
  const float scaler;
};

template <OperationType OpType> class FuseAttentionCTR {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

  using FuseAttentionCTRForwardParam =
      struct FuseAttentionCTRForwardParam<DataType_>;
  using FuseAttentionCTRBackwardParam =
      struct FuseAttentionCTRBackwardParam<DataType_>;

public:
  FuseAttentionCTR() {}

  unsigned long long cal_bufsize() { return 0; }

  void forward(FuseAttentionCTRForwardParam param);
  void backward(FuseAttentionCTRBackwardParam param);

  ~FuseAttentionCTR() {}
};
} // namespace fastertransformerv4
