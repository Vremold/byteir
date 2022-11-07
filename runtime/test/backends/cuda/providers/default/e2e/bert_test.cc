//===- bert_test.cc -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include "brt/core/common/status.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "brt/test/common/models.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <cuda_runtime.h>
#include <future>
#include <memory>
#include <string>

using namespace brt;
using namespace brt::cuda;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;
using namespace mlir;
using namespace std;

static std::string test_file_bert_tiny =
    "test/test_files/bert_tiny_host_cuda.mlir";

TEST(CUDATestE2E, BertTiny) {
  Session session;

  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load = session.Load(test_file_bert_tiny, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  // initialize index input
  for (auto offset : session.GetInputArgOffsets()) {
    void *data = request->GetArg(offset);
    auto dtype = session.GetDType(offset);
    if (dtype == DTypeEnum::Int64) { // for bert tiny index input
      RandCUDABuffer(
          reinterpret_cast<typename DTypeTraits<DTypeEnum::Int64>::type_t *>(
              data),
          LinearizedShape(session.GetStaticShape(offset)), 2);
    }
  }

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  // second
  status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
}

TEST(CUDATestE2E, BertTinyMultiGPUs) {
  int nr_device;
  BRT_CUDA_CHECK(cudaGetDeviceCount(&nr_device));
  if (!nr_device)
    return;

  auto worker = [](int device) -> void {
    BRT_CUDA_CHECK(cudaSetDevice(device));
    cudaStream_t stream_external;
    BRT_CUDA_CHECK(cudaStreamCreate(&stream_external));

    Session session;

    auto status_allocator = CUDAAllocatorFactory(&session, device);
    BRT_TEST_CHECK_STATUS(status_allocator);

    auto status_cuda = DefaultCUDAExecutionProviderFactory(&session, device);
    BRT_TEST_CHECK_STATUS(status_cuda);

    auto status_load = session.Load(test_file_bert_tiny, "byre");
    BRT_TEST_CHECK_STATUS(status_load);

    std::unique_ptr<RequestContext> request;
    auto status_request = session.NewRequestContext(&request);
    BRT_TEST_CHECK_STATUS(status_request);
    request->SetWorkQueue(new CUDAExternalStreamWorkQueue(stream_external));

    // initialize index input
    for (auto offset : session.GetInputArgOffsets()) {
      void *data = request->GetArg(offset);
      auto dtype = session.GetDType(offset);
      if (dtype == DTypeEnum::Int64) { // for bert tiny index input
        RandCUDABuffer(
            reinterpret_cast<typename DTypeTraits<DTypeEnum::Int64>::type_t *>(
                data),
            LinearizedShape(session.GetStaticShape(offset)), 2);
      }
    }

    request->FinishIOBinding();

    auto status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);

    // second
    status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);

    BRT_CUDA_CHECK(cudaStreamDestroy(stream_external));
  };

  std::vector<std::future<void>> futures;
  futures.reserve(nr_device);
  for (int i = 0; i < nr_device; ++i) {
    futures.emplace_back(std::async(std::launch::async, worker, i));
  }
  for (auto &&f : futures) {
    f.get();
  }
}
