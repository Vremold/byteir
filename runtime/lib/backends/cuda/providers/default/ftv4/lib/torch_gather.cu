/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v4/includes/reduce.h"
#include "fastertransformer_v4/includes/torch_gather.h"
#include "fastertransformer_v4/includes/utils.h"
using namespace std;

namespace fastertransformerv4 {
template <typename T, const int max_ite>
__global__ void
gather_torch_forward_kernel(const T *p2c, const T *c2p, const T *score,
                            T *final, const int batch_size, const int num_heads,
                            const int seq_len, const T scaler) {
  int bid = blockIdx.x;
  int offset = bid * seq_len * seq_len * 2;
  int score_offset = bid * seq_len * seq_len;

  const int max_len = 256;
  __shared__ float s_mem[32][max_len + 32];

  int warp_id = threadIdx.x >> 5;
  int warp_tid = threadIdx.x & 0x1F;

  for (int col_id = warp_id; col_id < (((seq_len + 31) >> 5) << 5);
       col_id += 32) // result col loop
  {
    int col_offset = col_id - warp_id;
    for (int i = 0; i < max_ite; i++) {
      if (col_id < seq_len)
        for (int tid = warp_tid + i * max_len;
             tid < min((i + 1) * max_len, seq_len) + 32; tid += 32) {
          int index = (seq_len - 1) + 32 + col_offset - tid;
          s_mem[warp_id][index % (max_len + 32)] =
              __ldg(&p2c[offset + col_id * (seq_len * 2) + index]);
        }

      __syncthreads();

      int warp_col_id = col_offset + warp_tid;
      if (warp_col_id < seq_len)
        for (int row_id = warp_id + i * max_len;
             row_id < min((i + 1) * max_len, seq_len);
             row_id += 32) // result row loop
        {
          int c2p_index = (seq_len - 1) + row_id - warp_col_id;
          T c2p_val = __ldg(&c2p[offset + row_id * (seq_len * 2) + c2p_index]);

          int p2c_index = (seq_len - 1) - row_id + warp_col_id;
          T p2c_val = s_mem[warp_tid][p2c_index % (max_len + 32)];

          int result_index = score_offset + row_id * seq_len + warp_col_id;
          final[result_index] =
              (p2c_val + c2p_val) * scaler + __ldg(&score[result_index]);
        }
      __syncthreads();
    }
  }
}

template <typename T, const int seq_len>
__global__ void gather_torch_backward_kernel(
    const T *grad_out, const T *grad_out_T, T *c2p_grad, T *p2c_grad,
    T *score_grad, const int batch_size, const int head_num, const T scaler) {
  int bid = blockIdx.x;
  int grad_offset = bid * seq_len * seq_len;

  const int ROW = 16;
  __shared__ float s_mem[ROW][seq_len + 1];
  __shared__ float s_mem2[ROW][seq_len + 1];

  for (int start_row = 0; start_row < seq_len; start_row += ROW) {
    int upper_bound = min(ROW, seq_len - start_row);
    for (int row_id = 0; row_id < upper_bound; row_id++) {
      int tid = threadIdx.x;
      int grad_id = grad_offset + (start_row + row_id) * seq_len + tid;

      T grad = __ldg(&grad_out[grad_id]);
      score_grad[grad_id] = grad;
      s_mem[row_id][tid] = grad * scaler;

      s_mem2[row_id][tid] = __ldg(&grad_out_T[grad_id]) * scaler;
    }

    __syncthreads();

    for (int row_id = 0; row_id < upper_bound; row_id++) {
      int c2p_grad_offset =
          (bid * seq_len + (start_row + row_id)) * (seq_len * 2);
      int p2c_grad_offset =
          (bid * seq_len + (start_row + row_id)) * (seq_len * 2);
      for (int tid = threadIdx.x; tid < seq_len * 2; tid += blockDim.x) {
        int col_id = (seq_len - 1) - (tid - (start_row + row_id));

        c2p_grad[c2p_grad_offset + tid] =
            (col_id >= 0 && col_id < seq_len) ? s_mem[row_id][col_id] : 0.0f;
        p2c_grad[p2c_grad_offset + tid] =
            (col_id >= 0 && col_id < seq_len) ? s_mem2[row_id][col_id] : 0.0f;
      }
    }

    __syncthreads();
  }
}

template <typename T>
__global__ void transpose_grad_kernel(const T *grad, T *gradT,
                                      const int batch_size, const int head_num,
                                      const int seq_len) {
  const int TILE_SIZE = 64;
  __shared__ T s_grad[TILE_SIZE][TILE_SIZE + 1];

  int bid = blockIdx.x;
  int offset = bid * seq_len * seq_len;

  for (int start_row = 0; start_row < seq_len; start_row += TILE_SIZE) {
    for (int start_col = 0; start_col < seq_len; start_col += TILE_SIZE) {
      for (int tid = threadIdx.x; tid < TILE_SIZE * TILE_SIZE;
           tid += blockDim.x) {
        int row_id = tid / TILE_SIZE;
        int col_id = tid % TILE_SIZE;
        s_grad[row_id][col_id] =
            __ldg(&grad[offset + (start_row + row_id) * seq_len + start_col +
                        col_id]);
      }

      __syncthreads();

      for (int tid = threadIdx.x; tid < TILE_SIZE * TILE_SIZE;
           tid += blockDim.x) {
        int row_id = tid / TILE_SIZE;
        int col_id = tid % TILE_SIZE;
        gradT[offset + (start_col + row_id) * seq_len + start_row + col_id] =
            s_grad[col_id][row_id];
      }

      __syncthreads();
    }
  }
}

template <OperationType OpType>
void TorchGather<OpType>::forward(TorchGatherForwardParam param) {
  dim3 grid(param.batch_size * param.head_num), block(1024);
  const int max_len = 256;
  const int max_ite = (param.seq_len + max_len - 1) / max_len;
  switch (max_ite) {
  case 1:
    gather_torch_forward_kernel<DataType_, 1><<<grid, block, 0, param.stream>>>(
        param.p2c_ptr, param.c2p_ptr, param.score_ptr, param.output,
        param.batch_size, param.head_num, param.seq_len,
        (DataType_)param.scaler);
    break;
  case 2:
    gather_torch_forward_kernel<DataType_, 2><<<grid, block, 0, param.stream>>>(
        param.p2c_ptr, param.c2p_ptr, param.score_ptr, param.output,
        param.batch_size, param.head_num, param.seq_len,
        (DataType_)param.scaler);
    break;
  case 3:
    gather_torch_forward_kernel<DataType_, 3><<<grid, block, 0, param.stream>>>(
        param.p2c_ptr, param.c2p_ptr, param.score_ptr, param.output,
        param.batch_size, param.head_num, param.seq_len,
        (DataType_)param.scaler);
    break;
  case 4:
    gather_torch_forward_kernel<DataType_, 4><<<grid, block, 0, param.stream>>>(
        param.p2c_ptr, param.c2p_ptr, param.score_ptr, param.output,
        param.batch_size, param.head_num, param.seq_len,
        (DataType_)param.scaler);
    break;
  default:
    printf("seq_len is larger than 1024\n");
  }
}

template <OperationType OpType>
void TorchGather<OpType>::backward(TorchGatherBackwardParam param) {
  dim3 grid(param.batch_size * param.head_num), block;

  DataType_ *gradT = param.grad_out_T;

  block.x = 512;
  transpose_grad_kernel<<<grid, block, 0, param.stream>>>(
      param.grad_out, gradT, param.batch_size, param.head_num, param.seq_len);

  block.x = param.seq_len;
  if (param.seq_len == 256)
    gather_torch_backward_kernel<DataType_, 256>
        <<<grid, block, 0, param.stream>>>(
            param.grad_out, gradT, param.c2p_grad, param.p2c_grad,
            param.score_grad, param.batch_size, param.head_num,
            (DataType_)param.scaler);
}

template void
TorchGather<OperationType::FP32>::forward(TorchGatherForwardParam param);
template void
TorchGather<OperationType::HALF>::forward(TorchGatherForwardParam param);

template void
TorchGather<OperationType::FP32>::backward(TorchGatherBackwardParam param);
template void
TorchGather<OperationType::HALF>::backward(TorchGatherBackwardParam param);
} // namespace fastertransformerv4