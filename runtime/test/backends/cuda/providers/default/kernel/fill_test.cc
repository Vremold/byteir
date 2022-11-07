//===- fill_test.cc -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "gtest/gtest.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static std::string test_file_fill = "test/test_files/fill_cuda.mlir";

using namespace brt;
using namespace brt::cuda;
using namespace brt::test;

TEST(CUDATestFillOp, Basic) {
  constexpr size_t length = 512 * 128;

  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load = session.Load(test_file_fill, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  CheckCUDAValues<float>(static_cast<float *>(request->GetArg(0)), length, 0.f);
  CheckCUDAValues<float>(static_cast<float *>(request->GetArg(1)), length, 1.f);
  CheckCUDAValues<__half>(static_cast<__half *>(request->GetArg(2)), length,
                          static_cast<__half>(1.f));
}
