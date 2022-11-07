//===- test_main.cc -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/core/common/common.h"
#include "brt/core/framework/execution_provider.h"
#include "brt/test/common/env.h"
#include "gtest/gtest.h"
#include <memory>

#define TEST_MAIN main

using namespace brt;
using namespace brt::test;

int TEST_MAIN(int argc, char **argv) {
  Env *env = Env::GetInstance(); // will creat a singleton
  BRT_UNUSED_PARAMETER(env);

  auto err = brt::ExecutionProvider::StaticRegisterKernelsFromDynlib(
      "lib/libexternal_kernels.so");
  BRT_ENFORCE(err.IsOK(), err.ErrorMessage());

  int status = 0;

  BRT_TRY {
    ::testing::InitGoogleTest(&argc, argv);
    status = RUN_ALL_TESTS();
  }
  BRT_CATCH(const std::exception &ex) {
    BRT_HANDLE_EXCEPTION([&]() {
      std::cerr << ex.what();
      status = -1;
    });
  }
  return status;
}
