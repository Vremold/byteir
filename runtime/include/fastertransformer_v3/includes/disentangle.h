/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once

#include "fastertransformer_v3/includes/common.h"

namespace fastertransformerv3 {
template <OperationType OpType> class DisentangleParam {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

public:
  int cublas_Algo[2];
  const DataType_ *relative_embedding;
  const DataType_ *attr_kernel_Q;
  const DataType_ *attr_kernel_K;
  bool is_paper;
  int scale;
  int max_pos;

  DisentangleParam() {
    relative_embedding = nullptr;
    attr_kernel_Q = nullptr;
    attr_kernel_K = nullptr;

    if (OpType == OperationType::HALF)
      cublas_Algo[0] = 99, cublas_Algo[1] = 99;
    else
      cublas_Algo[0] = -1, cublas_Algo[1] = -1;
  }
};

template <OperationType OpType> class Disentangle {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;
  DisentangleParam<OpType> param_;
  const int head_num_;
  const int size_per_head_;
  const int max_batch_size_;
  const int max_seq_len_;
  const int hidden_dim_;

public:
  Disentangle(const int head_num, const int size_per_head,
              const int max_batch_size, const int max_seq_len)
      : head_num_(head_num), size_per_head_(size_per_head),
        max_batch_size_(max_batch_size), max_seq_len_(max_seq_len),
        hidden_dim_(head_num * size_per_head) {}
  unsigned long long cal_bufsize() {
    return sizeof(DataType_) *
           (max_batch_size_ * max_seq_len_ * 2 * hidden_dim_ * 2 +
            max_seq_len_ * 2 * hidden_dim_ * 3 +
            max_batch_size_ * size_per_head_ * max_seq_len_ * max_seq_len_ * 2 *
                2);
  }
  void initialize(DisentangleParam<OpType> param) { param_ = param; }

  void infer(const DataType_ *attn_score, const DataType_ *query_out,
             const DataType_ *key_out, const DataType_ *query_bias,
             const DataType_ *key_bias, DataType_ *attn_score_out, void *buf,
             const int batch_size, const int seq_len,
             cublasHandle_t cublas_handle, cudaStream_t stream);

  ~Disentangle() {}
};
} // namespace fastertransformerv3
