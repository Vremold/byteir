/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v4/includes/dropout.h"
#include "fastertransformer_v4/includes/utils.h"

using namespace std;

namespace fastertransformerv4 {
template <typename T>
__global__ void dropout_fw_kernel(const T *in, uchar4 *mask, T *out,
                                  const int N, const float ratio,
                                  const int seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    float4 in4 = load_vector(in + tid * 4);
    float4 out4 = dropout_fw(in4, ratio, seed, tid, mask);
    store_vector(out + tid * 4, out4);
  }
}

template <typename T>
__global__ void dropout_bw_kernel(const T *in, const uchar4 *mask, T *out,
                                  const int N, const float scale) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid * 4 < N) {
    float4 in4 = load_vector(in + tid * 4);
    float4 out4 = dropout_bw(in4, scale, tid, mask);
    store_vector(out + tid * 4, out4);
  }
}

template <OperationType OpType>
void Dropout<OpType>::forward(DropoutForwardParam param) {
  const int seed = generate_random_seed();
  dim3 grid, block;
  block.x = 1024;
  grid.x = (param.N + (block.x * 4) - 1) / (block.x * 4);
  dropout_fw_kernel<<<grid, block, 0, param.stream>>>(
      param.dropout_in, param.dropout_mask, param.dropout_out, param.N, ratio_,
      seed);
}

template <OperationType OpType>
void Dropout<OpType>::backward(DropoutBackwardParam param) {
  dim3 grid, block;
  block.x = 1024;
  grid.x = (param.N + (block.x * 4) - 1) / (block.x * 4);
  dropout_bw_kernel<<<grid, block, 0, param.stream>>>(
      param.grad_out, param.dropout_mask, param.grad_in, param.N,
      1.0f / (1.0f - ratio_));
}

template void Dropout<OperationType::FP32>::forward(DropoutForwardParam param);
template void Dropout<OperationType::HALF>::forward(DropoutForwardParam param);

template void
Dropout<OperationType::FP32>::backward(DropoutBackwardParam param);
template void
Dropout<OperationType::HALF>::backward(DropoutBackwardParam param);
} // namespace fastertransformerv4