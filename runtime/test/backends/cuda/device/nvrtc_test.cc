//===- nvrtc_test.cc ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/compile/nvrtc.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/test/common/cuda/util.h"
#include "test_kernels.h"
#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <string>

using namespace brt;
using namespace brt::cuda;
using namespace brt::test;

static void CheckResult(float *d_ptr, size_t size, float val) {
  CheckCUDABuffer<float>(d_ptr, size, [&](float *h_ptr) {
    for (size_t i = 0; i < size; ++i) {
      EXPECT_NEAR(h_ptr[i], val, 1e-6f);
    }
  });
}

static std::string test_file_nvrtc = "test/test_files/cuda_add.cu";
static std::string test_file_nvrtc_kerenl = "nvrtc_add_kernel";

TEST(NVRTCTest, Add) {

  CUDARTCompilation *nvrtc_handle = CUDARTCompilation::GetInstance();

  CUfunction func;

  auto status_nvrtc = nvrtc_handle->GetOrCreateFunction(
      func, test_file_nvrtc_kerenl, 0, test_file_nvrtc);

  BRT_TEST_CHECK_STATUS(status_nvrtc);

  CUDAWorkQueue wq(0);

  int gx = 4;
  int bx = 256;

  dim3 grid(gx, 1, 1);
  dim3 block(bx, 1, 1);
  size_t shared_size = 0;

  float *arr1;
  float *arr2;
  float *arr3;
  int n = gx * bx;
  float val1 = 1.0f;
  float val2 = 2.0f;

  size_t count = n * sizeof(float);

  BRT_CUDA_CHECK(cudaMalloc(&arr1, count));
  BRT_CUDA_CHECK(cudaMalloc(&arr2, count));
  BRT_CUDA_CHECK(cudaMalloc(&arr3, count));

  BRT_CUDA_CHECK(cudaMemset(arr1, 0, count));
  BRT_CUDA_CHECK(cudaMemset(arr2, -1, count));
  BRT_CUDA_CHECK(cudaMemset(arr3, -1, count));
  cudaDeviceSynchronize();

  void *args1[] = {&grid, &block, &shared_size, &arr1, &arr2, &n, &val1};
  wq.AddTask(5, (void *)func, args1);

  void *args2[] = {&grid, &block, &shared_size, &arr2, &arr3, &n, &val2};
  wq.AddTask(5, (void *)func, args2);

  wq.Sync();

  CheckResult(arr3, n, 3.0f);

  BRT_CUDA_CHECK(cudaFree(arr1));
  BRT_CUDA_CHECK(cudaFree(arr2));
  BRT_CUDA_CHECK(cudaFree(arr3));
}
