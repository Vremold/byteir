//===- copy.cc -------------------------------------------------*--- C++
//-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "./copy.h"

#include "brt/backends/cuda/device/compile/ptx.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/ir/engine_util.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/IR/BuiltinOps.h" // ModuleOp
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

using namespace brt;
using namespace brt::common;
using namespace brt::cuda;
using namespace brt::ir;
using namespace llvm;
using namespace mlir;

namespace brt {
namespace cuda {

CopyOpKernel::CopyOpKernel(const OpKernelInfo &info, int type)
    : OpKernel(info), task_type(type) {
  src_id = GetTensorIndexFromOpArgIndex(info_, 0);
  dst_id = GetTensorIndexFromOpArgIndex(info_, 1);

  // get static bytes
  // TODO change to dynamic later
  auto src_val = GetMLIRValueFromOpArgIndex(info_, 0);
  auto maybe_bytes = GetStaticBytes(src_val);
  if (maybe_bytes.has_value()) {
    byte_size = maybe_bytes.value();
  }
}

CopyOpKernel::~CopyOpKernel() {}

common::Status CopyOpKernel::RunImpl(const ExecutionContext &ctx) {
  std::vector<void *> args(3);
  AsyncValueRef dst_value = ctx.exec_frame->GetAsyncValueRef(dst_id);
  AsyncValueRef src_value = ctx.exec_frame->GetAsyncValueRef(src_id);
  args[0] = &dst_value;
  args[1] = &src_value;
  args[2] = &byte_size;
  auto work_queue = static_cast<CUDAWorkQueue *>(ctx.work_queue);
  return work_queue->AddTask(task_type, nullptr, args.data());
}

} // namespace cuda
} // namespace brt
