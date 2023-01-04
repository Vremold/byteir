//===- fast_transformer_test.cc -------------------------------*--- C++ -*-===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

// TODO remove this before opnesource

#include "backends/cuda/providers/default/ftv4/models.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include "brt/core/common/status.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <string>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test::cuda;

class CUDATestFastTransformerSingleOp : public testing::Test {
public:
  template <auto builder> void Run() {
    Session session;
    auto status_allocator = CUDAAllocatorFactory(&session);
    BRT_TEST_CHECK_STATUS(status_allocator);
    auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
    BRT_TEST_CHECK_STATUS(status_cuda);

    ByREBuilder byre_builder;
    auto status_load = session.LoadFromMemory(builder(byre_builder), "byre");
    BRT_TEST_CHECK_STATUS(status_load);

    std::unique_ptr<RequestContext> request;
    auto status_request = session.NewRequestContext(&request);
    BRT_TEST_CHECK_STATUS(status_request);

    request->FinishIOBinding();

    auto status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);

    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);
  }
};

TEST_F(CUDATestFastTransformerSingleOp, LayerNorm) {
  Run<ftv4::CreateLayerNorm>();
}

TEST_F(CUDATestFastTransformerSingleOp, LayerNormBackward) {
  Run<ftv4::CreateLayerNormBackward>();
}

TEST_F(CUDATestFastTransformerSingleOp, Softmax) { Run<ftv4::CreateSoftmax>(); }

TEST_F(CUDATestFastTransformerSingleOp, SoftmaxBackward) {
  Run<ftv4::CreateSoftmaxBackward>();
}

TEST_F(CUDATestFastTransformerSingleOp, LinearGeluDropout) {
  Run<ftv4::CreateLinearGeluDropout>();
}

TEST_F(CUDATestFastTransformerSingleOp, LinearGeluDropoutBackward) {
  Run<ftv4::CreateLinearGeluDropoutBackward>();
}

TEST_F(CUDATestFastTransformerSingleOp, LinearTranspose0213) {
  Run<ftv4::CreateLinearTranspose0213>();
}

TEST_F(CUDATestFastTransformerSingleOp, LinearTranspose0213Backward) {
  Run<ftv4::CreateLinearTranspose0213Backward>();
}

TEST_F(CUDATestFastTransformerSingleOp, Transpose4d2013) {
  Run<ftv4::CreateTranspose4d2013>();
}

TEST_F(CUDATestFastTransformerSingleOp, Matmul) { Run<ftv4::CreateMatmul>(); }
