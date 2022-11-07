//===- string_equal_test.cc -----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/backends/cpu/providers/default/cpu_provider.h"
#include "brt/core/common/status.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/util.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <future>
#include <memory>
#include <string>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;
using namespace mlir;
using namespace std;

// module which compares input strings with "aaa"s
static std::string test_file_string_equal = "test/test_files/string_equal.mlir";
static std::string test_file_string_equal_scalar =
    "test/test_files/string_equal_scalar.mlir";

TEST(CPUTestE2E, StringEqual) {
  Session session;

  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.Load(test_file_string_equal, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  StringView *src = reinterpret_cast<StringView *>(request->GetArg(0));
  bool *dest = reinterpret_cast<bool *>(request->GetArg(1));

  request->FinishIOBinding();

  for (size_t i = 0; i < 2; ++i) {
    src[0] = "aa";
    src[1] = "aaa";
    src[2] = "abc";
    src[3] = "aaa";

    auto status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);

    // check results
    ASSERT_FALSE(dest[0]);
    ASSERT_TRUE(dest[1]);
    ASSERT_FALSE(dest[2]);
    ASSERT_TRUE(dest[3]);
  }
}

TEST(CPUTestE2E, StringEqualScalar) {
  Session session;

  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.Load(test_file_string_equal_scalar, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  StringView *src = reinterpret_cast<StringView *>(request->GetArg(0));
  bool *dest = reinterpret_cast<bool *>(request->GetArg(1));

  request->FinishIOBinding();

  for (size_t i = 0; i < 2; ++i) {
    src[0] = "aaa";

    auto status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);

    // check results
    ASSERT_TRUE(dest[0]);
    ASSERT_TRUE(dest[1]);
    ASSERT_TRUE(dest[2]);
    ASSERT_TRUE(dest[3]);

    src[0] = "aa";

    status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);

    // check results
    ASSERT_FALSE(dest[0]);
    ASSERT_FALSE(dest[1]);
    ASSERT_FALSE(dest[2]);
    ASSERT_FALSE(dest[3]);
  }
}
