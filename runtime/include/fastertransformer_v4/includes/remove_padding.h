/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "common.h"

namespace fastertransformerv4 {
template <typename T> struct RemovePaddingParam {
  const T *input;
  const int *word_idx;
  T *output;
  const int batch_size;
  const int seq_len;
  const int hidden_dim;
  cudaStream_t stream;
  int valid_word_num = 0;
};

template <typename T> struct GetValidWordIndexParam {
  const T *attention_mask;
  int *word_idx;
  const int batch_size;
  const int seq_len;
  cudaStream_t stream;
  int *h_valid_word_num_ptr;
};

template <OperationType OpType> class RemovePadding {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

  using RemovePaddingParam = struct RemovePaddingParam<DataType_>;
  using GetValidWordIndexParam = struct GetValidWordIndexParam<DataType_>;

public:
  RemovePadding() {}

  static void compress(RemovePaddingParam param);
  static void restore(RemovePaddingParam param);
  static void get_valid_word_index(GetValidWordIndexParam param);

  ~RemovePadding() {}
};
} // namespace fastertransformerv4