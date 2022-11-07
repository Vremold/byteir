//===- allocator_test.cc --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/core/framework/allocator.h"
#include "brt/core/framework/arena.h"
#include "brt/core/framework/bfc_arena.h"
#include "gtest/gtest.h"

#include "brt/test/common/config.h"
#include "brt/test/common/util.h"
#include <string>

using namespace brt;
using namespace brt::test;

static void CheckResult(void *ptr, size_t size, char val) {
  CheckValues<char>((char *)ptr, size, val);
}

static inline void test_func(IAllocator *cpu_allocator) {
#if BRT_TEST_WITH_ASAN
  size_t large_size = 32 * 1024 * 1024;
#else
  size_t large_size = 1024 * 1024 * 1024;
#endif

  auto ptr = cpu_allocator->Alloc(large_size);
  EXPECT_TRUE(ptr != nullptr);
  // test the bytes are ok for read/write
  memset(ptr, -1, large_size);

  CheckResult(ptr, large_size, -1);
  cpu_allocator->Free(ptr);

  size_t small_size = 1024 * 1024; // 1MB
  size_t cnt = 1024;
  // check time for many allocation
  for (size_t s = 0; s < cnt; ++s) {
    void *raw = cpu_allocator->Alloc(small_size);
    cpu_allocator->Free(raw);
  }
  EXPECT_TRUE(true);
}

TEST(AllocatorTest, CPUBase) {
  CPUAllocator cpu_base_allocator;
  test_func(&cpu_base_allocator);
}

TEST(AllocatorTest, CPUArena) {
  BFCArena cpu_bfc_allocator(std::unique_ptr<IAllocator>(new CPUAllocator()),
                             1 << 30);
  test_func(&cpu_bfc_allocator);
}
