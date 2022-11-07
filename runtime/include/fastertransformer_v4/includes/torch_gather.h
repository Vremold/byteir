/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "common.h"

namespace fastertransformerv4 {
template <typename T> struct TorchGatherForwardParam {
  const T *c2p_ptr;
  const T *p2c_ptr;
  const T *score_ptr;
  T *output;
  void *buf;
  const float scaler;
  const int batch_size;
  const int head_num;
  const int seq_len;
  cudaStream_t stream;
};

template <typename T> struct TorchGatherBackwardParam {
  const T *grad_out;
  T *grad_out_T;
  T *c2p_grad;
  T *p2c_grad;
  T *score_grad;
  void *buf;
  const float scaler;
  const int batch_size;
  const int head_num;
  const int seq_len;
  cudaStream_t stream;
};

template <OperationType OpType> class TorchGather {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

  using TorchGatherForwardParam = struct TorchGatherForwardParam<DataType_>;
  using TorchGatherBackwardParam = struct TorchGatherBackwardParam<DataType_>;

public:
  TorchGather() {}

  unsigned long long cal_bufsize() { return 0; }

  void forward(TorchGatherForwardParam param);

  void backward(TorchGatherBackwardParam param);

  ~TorchGather() {}
};
} // namespace fastertransformerv4
