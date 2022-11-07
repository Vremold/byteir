/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "common.h"
#include <chrono>
#include <curand_kernel.h>

namespace fastertransformerv4 {
typedef union half4 {
  float2 x;
  __half h[4];
} half4;

__inline__ __device__ float4 load_vector(const float *ptr) {
  return *(const float4 *)ptr;
}

__inline__ __device__ float4 load_vector(const __half *ptr) {
  half4 tmp;
  tmp.x = *(const float2 *)ptr;
  return make_float4((float)tmp.h[0], (float)tmp.h[1], (float)tmp.h[2],
                     (float)tmp.h[3]);
}

__inline__ __device__ void store_vector(float *ptr, float4 x) {
  *(float4 *)ptr = x;
}

__inline__ __device__ void store_vector(__half *ptr, float4 x) {
  half4 tmp;
  tmp.h[0] = (__half)x.x, tmp.h[1] = (__half)x.y, tmp.h[2] = (__half)x.z,
  tmp.h[3] = (__half)x.w;
  *(float2 *)ptr = tmp.x;
}

__inline__ int generate_random_seed() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

__inline__ __device__ float dropout_fw(float in, const float ratio,
                                       const int seed, const int tid,
                                       uint8_t *mask) {
  curandStatePhilox4_32_10_t state;
  curand_init(seed, tid, 0, &state);
  float rand = curand_uniform(&state);

  uint8_t mask_ = rand > ratio ? 1 : 0;
  mask[tid] = mask_;

  const float scale = __fdividef(1.0f, 1.0f - ratio);
  return in * scale * (float)mask_;
}

__inline__ __device__ half2 dropout_fw(half2 in, const float ratio,
                                       const int seed, const int tid,
                                       uchar2 *mask) {
  curandStatePhilox4_32_10_t state;
  curand_init(seed, tid, 0, &state);
  float2 rand;
  rand.x = curand_uniform(&state);
  rand.y = curand_uniform(&state);

  uchar2 mask2 = make_uchar2(rand.x > ratio, rand.y > ratio);
  mask[tid] = mask2;

  const float scale = __fdividef(1.0f, 1.0f - ratio);
  float2 out2;
  out2.x = (float)in.x * scale * (float)mask2.x;
  out2.y = (float)in.y * scale * (float)mask2.y;

  return __float22half2_rn(out2);
}

__inline__ __device__ float4 dropout_fw(float4 in4, const float ratio,
                                        const int seed, const int tid,
                                        uchar4 *mask) {
  curandStatePhilox4_32_10_t state;
  curand_init(seed, tid, 0, &state);
  float4 rand = curand_uniform4(&state);

  uchar4 mask4 = make_uchar4(rand.x > ratio, rand.y > ratio, rand.z > ratio,
                             rand.w > ratio);
  mask[tid] = mask4;

  const float scale = __fdividef(1.0f, 1.0f - ratio);
  float4 out4;
  out4.x = in4.x * scale * (float)mask4.x;
  out4.y = in4.y * scale * (float)mask4.y;
  out4.z = in4.z * scale * (float)mask4.z;
  out4.w = in4.w * scale * (float)mask4.w;

  return out4;
}

__inline__ __device__ float dropout_bw(float in, float scale, const int tid,
                                       const uint8_t *mask) {
  return in * scale * (float)mask[tid];
}

__inline__ __device__ half2 dropout_bw(half2 in, float scale, const int tid,
                                       const uchar2 *mask) {
  uchar2 mask2 = mask[tid];
  float2 out2;
  out2.x = (float)in.x * scale * (float)mask2.x;
  out2.y = (float)in.y * scale * (float)mask2.y;

  return __float22half2_rn(out2);
}

__inline__ __device__ float4 dropout_bw(float4 in4, float scale, const int tid,
                                        const uchar4 *mask) {
  uchar4 mask4 = mask[tid];
  float4 out4;
  out4.x = in4.x * scale * (float)mask4.x;
  out4.y = in4.y * scale * (float)mask4.y;
  out4.z = in4.z * scale * (float)mask4.z;
  out4.w = in4.w * scale * (float)mask4.w;

  return out4;
}

__inline__ __device__ float gelu_fw(float x) {
  float cdf =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

__inline__ __device__ float gelu_bw(float grad, float x) {
  const float sqrt_param = 0.79788456080286535587989211986876f;
  float x2mul = x * x * 0.044715f;
  float tan_h = tanhf(sqrt_param * (x + x * x2mul));
  float dg1 = 0.5f * (1.0f + tan_h);
  float dg2 = x * 0.5f * sqrt_param * (1 - tan_h * tan_h);
  float dg3 = dg2 * 3 * x2mul;
  return grad * (dg1 + dg2 + dg3);
}

// template <ActType act, typename T>
// __global__
// void add_bias_act(T *output, const T *bias, const int M, const int N)
// {
//     int row_offset = blockIdx.x * N;
//     for(int tid = threadIdx.x; tid < N; tid += blockDim.x)
//     {
//         T out = output[row_offset + tid] + __ldg(&bias[tid]);
//         out = act_fun<act>(out);
//         output[row_offset + tid] = out;
//     }
// }
} // namespace fastertransformerv4