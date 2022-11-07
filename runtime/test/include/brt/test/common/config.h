//===- config.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

// clang-format off
#if defined(__has_feature)
    #if __has_feature(address_sanitizer)
        #define BRT_TEST_WITH_ASAN 1
    #else
        #define BRT_TEST_WITH_ASAN 0
    #endif
#elif defined(__SANITIZE_ADDRESS__)
    #define BRT_TEST_WITH_ASAN 1
#else
    #define BRT_TEST_WITH_ASAN 0
#endif
// clang-format on
