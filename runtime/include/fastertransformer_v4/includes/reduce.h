/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "common.h"

namespace fastertransformerv4 {
#define FINAL_MASK 0xffffffff

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

  return wid == 0 ? warpReduceSum(threadIdx.x < (blockDim.x >> 5) ? shared[lane]
                                                                  : (T)0.0f)
                  : 0.0f;
}

__inline__ __device__ __half2 warpReduceSum(__half2 val) {
  half2 tmp_val;
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp_val = __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    val = __hadd2(tmp_val, val);
  }
  return val;
}

__inline__ __device__ __half __half2add(__half2 val) {
  return __hadd(val.x, val.y);
}

__inline__ __device__ __half blockReduceSum(__half2 val) {
  static __shared__ __half2 shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<__half2>(val);

  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  return (__half)(wid == 0 ? warpReduceSum(threadIdx.x < (blockDim.x >> 5)
                                               ? (float)__half2add(shared[lane])
                                               : 0.0f)
                           : 0.0f);
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

  return wid == 0 ? warpReduceMax(threadIdx.x < (blockDim.x >> 5) ? shared[lane]
                                                                  : -1e20f)
                  : -1e20f;
}

__inline__ __device__ float2 warpReduceSum_2(float2 val) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    val.x += __shfl_xor_sync(FINAL_MASK, val.x, mask, 32);
    val.y += __shfl_xor_sync(FINAL_MASK, val.y, mask, 32);
  }
  return val;
}

__inline__ __device__ float2 blockReduceSum_2(float2 val) {
  static __shared__ float shared[2][32];

  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum_2(val);

  if (lane == 0) {
    shared[0][wid] = val.x;
    shared[1][wid] = val.y;
  }
  __syncthreads();

  return wid == 0 ? warpReduceSum_2(
                        threadIdx.x < (blockDim.x >> 5)
                            ? make_float2(shared[0][lane], shared[1][lane])
                            : make_float2(0.0f, 0.0f))
                  : make_float2(0.0f, 0.0f);
}
} // namespace fastertransformerv4