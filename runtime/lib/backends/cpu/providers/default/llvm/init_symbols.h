//===- init_symbols.h -----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

namespace brt {
namespace cpu {
class LLVMJIT;

void InitJITKernelRTSymbols(LLVMJIT *jit);

} // namespace cpu
} // namespace brt
