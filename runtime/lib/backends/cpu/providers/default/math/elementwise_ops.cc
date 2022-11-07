//===- elementwise_ops.cc -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/backends/cpu/providers/default/math/elementwise_ops.h"

#include "brt/core/common/common.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include <iostream>

using namespace brt;
using namespace brt::common;
using namespace mlir;

namespace brt {
namespace cpu {

template <typename T>
common::Status Add<T>::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  auto N = accessor.GetNumElementsOfShape(accessor.GetArgShape(0));
  T *a = static_cast<T *>(accessor.GetArgAsyncValueRef(0));
  T *b = static_cast<T *>(accessor.GetArgAsyncValueRef(1));
  T *c = static_cast<T *>(accessor.GetArgAsyncValueRef(2));
  for (unsigned int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }

  return Status::OK();
}

// instantiate
template class Add<float>;
template class Add<int>;

} // namespace cpu
} // namespace brt
