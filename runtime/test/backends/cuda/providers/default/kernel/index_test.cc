//===- index_test.cc ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include "brt/core/common/status.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "brt/test/common/models.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <string>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;

namespace {
size_t getNumElementsOfShape(const std::vector<int64_t> &shape) {
  size_t ret = 1;
  for (auto &&i : shape) {
    ret *= i;
  }
  return ret;
}

void CheckAliasThenIndexPutFisrtDimCUDAValue(
    void *d_input, void *d_index, void *d_update, void *d_output,
    const std::vector<int64_t> &inout_shape,
    const std::vector<int64_t> &index_shape, float eps) {

  int index_bound = index_shape[0];
  size_t update_size = index_bound;
  int feature_size = 1;
  for (size_t i = 1; i < inout_shape.size(); ++i) {
    feature_size *= inout_shape[i];
  }

  update_size *= feature_size;

  size_t inout_size = getNumElementsOfShape(inout_shape);
  size_t inout_size_in_bytes = inout_size * sizeof(float);

  size_t index_size_in_bytes =
      getNumElementsOfShape(index_shape) * sizeof(int64_t);
  size_t update_size_in_bytes = update_size * sizeof(float);

  float *h_input = (float *)malloc(inout_size_in_bytes);
  int64_t *h_index = (int64_t *)malloc(index_size_in_bytes);
  float *h_update = (float *)malloc(update_size_in_bytes);
  float *h_output = (float *)malloc(inout_size_in_bytes);

  cudaMemcpy(h_input, d_input, inout_size_in_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_index, d_index, index_size_in_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_update, d_update, update_size_in_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output, d_output, inout_size_in_bytes, cudaMemcpyDeviceToHost);

  std::vector<float> result(inout_size);
  std::copy(h_input, h_input + inout_size, result.begin());

  for (int i = 0; i < index_bound; ++i) {
    int in_offset = i * feature_size;
    int out_offset = h_index[i] * feature_size;
    for (int j = 0; j < feature_size; ++j) {
      int out_index = out_offset + j;
      int in_index = in_offset + j;
      result[out_index] += h_update[in_index];
    }
  }

  for (size_t i = 0; i < inout_size; ++i) {
    EXPECT_NEAR(result[i], h_output[i], eps);
  }

  free(h_input);
  free(h_index);
  free(h_update);
  free(h_output);
}

void CheckIndexPutFirstDim(const std::vector<int64_t> &inout_shape,
                           const std::vector<int64_t> &index_shape) {

  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  ByREBuilder byre_builder;
  auto status_load = session.LoadFromMemory(
      CreateIndexPut(byre_builder, "cuda", inout_shape, 0 /*dim*/, index_shape),
      "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  size_t update_size = index_shape[0];
  int feature_size = 1;
  for (size_t i = 1; i < inout_shape.size(); ++i) {
    feature_size *= inout_shape[i];
  }
  update_size *= feature_size;

  // initiate input
  RandCUDABuffer((float *)request->GetArg(0),
                 getNumElementsOfShape(inout_shape));
  // initiate index
  RandCUDABuffer((int64_t *)request->GetArg(1),
                 getNumElementsOfShape(index_shape), inout_shape[0]);
  // initiate update
  RandCUDABuffer((float *)request->GetArg(2), update_size);

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  float eps = 1e-5f;

  CheckAliasThenIndexPutFisrtDimCUDAValue(
      request->GetArg(0), request->GetArg(1), request->GetArg(2),
      request->GetArg(3), inout_shape, index_shape, eps);

  // second run
  RandCUDABuffer((float *)request->GetArg(0),
                 getNumElementsOfShape(inout_shape));
  RandCUDABuffer((int64_t *)request->GetArg(1),
                 getNumElementsOfShape(index_shape), inout_shape[0]);
  RandCUDABuffer((float *)request->GetArg(2), update_size);

  status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  CheckAliasThenIndexPutFisrtDimCUDAValue(
      request->GetArg(0), request->GetArg(1), request->GetArg(2),
      request->GetArg(3), inout_shape, index_shape, eps);
}

void CheckIndexSelectHost(void *d_input, void *d_index, void *d_output,
                          const std::vector<int64_t> &input_shape, size_t dim,
                          const std::vector<int64_t> &index_shape, float eps) {
  std::vector<int64_t> output_shape = input_shape;
  output_shape[dim] = index_shape[0];

  size_t input_size_in_bytes =
      getNumElementsOfShape(input_shape) * sizeof(float);
  size_t index_size_in_bytes =
      getNumElementsOfShape(index_shape) * sizeof(uint32_t);
  size_t output_size = getNumElementsOfShape(output_shape);
  size_t output_size_in_bytes = output_size * sizeof(float);

  float *h_input = (float *)malloc(input_size_in_bytes);
  uint32_t *h_index = (uint32_t *)malloc(index_size_in_bytes);
  float *h_output = (float *)malloc(output_size_in_bytes);

  cudaMemcpy(h_input, d_input, input_size_in_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_index, d_index, index_size_in_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output, d_output, output_size_in_bytes, cudaMemcpyDeviceToHost);
  std::vector<float> result(output_size);

  std::function<void(size_t, size_t, size_t)> f;
  f = [&](size_t cur_dim, size_t i_offset, size_t o_offset) {
    if (cur_dim == input_shape.size()) {
      result[o_offset] = h_input[i_offset];
      return;
    }
    if (cur_dim != dim) {
      for (int64_t i = 0; i < input_shape[cur_dim]; ++i) {
        f(cur_dim + 1, i_offset * input_shape[cur_dim] + i,
          o_offset * output_shape[cur_dim] + i);
      }
    } else {
      for (int64_t i = 0; i < index_shape[0]; ++i) {
        f(cur_dim + 1, i_offset * input_shape[cur_dim] + h_index[i],
          o_offset * output_shape[cur_dim] + i);
      }
    }
  };
  f(0, 0, 0);

  for (size_t i = 0; i < output_size; ++i) {
    EXPECT_NEAR(result[i], h_output[i], eps);
  }

  free(h_input);
  free(h_index);
  free(h_output);
}

void CheckIndexSelectSingle(const std::vector<int64_t> &input_shape, size_t dim,
                            const std::vector<int64_t> &index_shape) {
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  ByREBuilder byre_builder;
  auto status_load = session.LoadFromMemory(
      CreateIndexSelect(byre_builder, "cuda", input_shape, dim, index_shape),
      "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  // initiate input
  RandCUDABuffer((float *)request->GetArg(0),
                 getNumElementsOfShape(input_shape));
  // initiate index
  RandCUDABuffer((uint32_t *)request->GetArg(1),
                 getNumElementsOfShape(index_shape), input_shape[dim]);

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  CheckIndexSelectHost(request->GetArg(0), request->GetArg(1),
                       request->GetArg(2), input_shape, dim, index_shape,
                       1e-8f);
  // second run
  RandCUDABuffer((float *)request->GetArg(0),
                 getNumElementsOfShape(input_shape));
  RandCUDABuffer((uint32_t *)request->GetArg(1),
                 getNumElementsOfShape(index_shape), input_shape[dim]);
  status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  CheckIndexSelectHost(request->GetArg(0), request->GetArg(1),
                       request->GetArg(2), input_shape, dim, index_shape,
                       1e-8f);
}
} // namespace

TEST(CUDATestIndexOp, IndexPut) {
  CheckIndexPutFirstDim({3, 2}, {5});
  CheckIndexPutFirstDim({256, 128}, {128});
  CheckIndexPutFirstDim({256, 128}, {32});
  CheckIndexPutFirstDim({256, 128}, {64});
}

TEST(CUDATestIndexOp, IndexPutLarge) {
  CheckIndexPutFirstDim({30522, 128}, {128});
}

TEST(CUDATestIndexOp, IndexSelect) {
  CheckIndexSelectSingle({2, 3, 4, 5}, 0, {2});
  CheckIndexSelectSingle({2, 3, 4, 5}, 1, {4});
  CheckIndexSelectSingle({2, 3, 4, 5}, 2, {1});
  CheckIndexSelectSingle({2, 3, 4, 5}, 3, {3});
  CheckIndexSelectSingle({256, 64, 128}, 0, {128});
  CheckIndexSelectSingle({256, 64, 128}, 1, {32});
  CheckIndexSelectSingle({256, 64, 128}, 2, {64});
  CheckIndexSelectSingle({30522, 128}, 0, {128});
}