/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "fastertransformer_v3/includes/common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformerv3 {
#define FINAL_MASK 0xffffffff
#define PI 3.141592654f

template <typename T> __inline__ __device__ T warpReduceSum(T val) {
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T> __inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}

__inline__ __device__ __half2 warpReduceSum(__half2 val) {
  half2 tmp_val;
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp_val = __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    val = __hadd2(tmp_val, val);
  }
  return val;
}

__inline__ __device__ __half blockReduceSum(__half2 val) {
  static __shared__ __half2 shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<__half2>(val);

  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane]
                                          : __half2half2((__half)0.0f);
  val = warpReduceSum(val);
  return __hadd(val.x, val.y);
}

template <typename T> __inline__ __device__ T warpReduceMax(T val) {
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

template <typename T> __inline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax(val);

  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);
  return val;
}

template <ActType act, typename T> __inline__ __device__ T act_fun(T val) {
  if (act == ActType::Relu)
    return (val <= (T)0.0f) ? (T)0.0f : val;
  else if (act == ActType::SoftPlus)
    return logf(__expf((float)val) + 1.0f);
  else if (act == ActType::Sigmoid)
    return 1.0f / (1.0f + __expf(-1.0f * (float)val));
  else
    return val;
}

} // namespace fastertransformerv3