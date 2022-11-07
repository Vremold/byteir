/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v4/includes/transpose.h"
using namespace std;

namespace fastertransformerv4 {
const int WARP_SIZE = 32;

template <transposeType transpose_type>
__global__ void transpose4d_kernel(const float *input, float *output,
                                   const int dim_1, const int dim_2,
                                   const int dim_3, const int dim_4) {
  int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
  int total_data_blocks = dim_1 * dim_2 * dim_3;
  int total_warps = blockDim.x * gridDim.x / WARP_SIZE;

  int local_tid = threadIdx.x % WARP_SIZE;

  for (; warp_id < total_data_blocks; warp_id += total_warps) {
    int d1 = warp_id / (dim_2 * dim_3);
    int d2 = (warp_id % (dim_2 * dim_3)) / dim_3;
    int d3 = warp_id % dim_3;
    int source_row_id = warp_id;
    int target_row_id =
        transpose3d<transpose_type>(dim_1, dim_2, dim_3, d1, d2, d3);
    for (int tid = local_tid; tid < dim_4; tid += WARP_SIZE)
      output[target_row_id * dim_4 + tid] = input[source_row_id * dim_4 + tid];
  }
}

template <transposeType transpose_type>
__global__ void transpose4d_kernel(const __half *input, __half *output,
                                   const int dim_1, const int dim_2,
                                   const int dim_3, const int dim_4) {
  int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
  int total_data_blocks = dim_1 * dim_2 * dim_3;
  int total_warps = blockDim.x * gridDim.x / WARP_SIZE;

  int local_tid = threadIdx.x % WARP_SIZE;

  const half2 *input_ptr = (const half2 *)(input);
  half2 *output_ptr = (half2 *)(output);
  int dim_4_half = dim_4 / 2;

  for (; warp_id < total_data_blocks; warp_id += total_warps) {
    int d1 = warp_id / (dim_2 * dim_3);
    int d2 = (warp_id % (dim_2 * dim_3)) / dim_3;
    int d3 = warp_id % dim_3;
    int source_row_id = warp_id;
    int target_row_id =
        transpose3d<transpose_type>(dim_1, dim_2, dim_3, d1, d2, d3);
    for (int tid = local_tid; tid < dim_4_half; tid += WARP_SIZE)
      output_ptr[target_row_id * dim_4_half + tid] =
          input_ptr[source_row_id * dim_4_half + tid];
  }
}

template <OperationType OpType>
void Transpose<OpType>::forward(TransposeParam param) {
  dim3 grid(320), block(512);
  switch (param.transpose_type) {
  case TRANSPOSE0213:
    transpose4d_kernel<TRANSPOSE0213><<<grid, block, 0, param.stream>>>(
        param.input, param.output, param.dim_1, param.dim_2, param.dim_3,
        param.dim_4);
    break;
  case TRANSPOSE1203:
    transpose4d_kernel<TRANSPOSE1203><<<grid, block, 0, param.stream>>>(
        param.input, param.output, param.dim_1, param.dim_2, param.dim_3,
        param.dim_4);
    break;
  case TRANSPOSE2013:
    transpose4d_kernel<TRANSPOSE2013><<<grid, block, 0, param.stream>>>(
        param.input, param.output, param.dim_1, param.dim_2, param.dim_3,
        param.dim_4);
    break;
  }
}

template void Transpose<OperationType::FP32>::forward(TransposeParam param);
template void Transpose<OperationType::HALF>::forward(TransposeParam param);
} // namespace fastertransformerv4