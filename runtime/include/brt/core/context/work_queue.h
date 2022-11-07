//===- work_queue.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/common/status.h"
#include <functional>
#include <string>

namespace brt {

/**
 * WorkQueue is an abstract object holding scheduled tasks.
 * Tasks in a WorkQueue may or may not execute sequentially,
 * depending on the implementation of derived version.
 *
 * E.g. A derived WorkQueue, denoted CUDAStreamWorkQueue,
 * can be implemented through a CUDA stream
 * that maintain seqeuntial order within a CUDA stream.
 *
 * E.g. A derived WorkQueue, denoted CUDAMultiStreamWorkQueue,
 * can be implemented through multiple CUDA streams
 * that can concurrently run data trasnfer and computation.
 *
 * E.g. A derived WorkQueue, denoted CPUSingleThreadWorkQueue,
 * can be implemented using single thread that run sequentially.
 *
 * E.g. A derived WorkQueue, denoted CPUMultiThreadWorkQueue,
 * can be implemented using multiple threads
 * that allow multiple worker threads.
 */

class WorkQueue {
public:
  WorkQueue(const std::string &name) : name_(name){};

  // Undefined what happens to pending work when destructor is called.
  virtual ~WorkQueue() {}

  // Return a human-readable description of the work queue.
  const std::string &name() const { return name_; }

  // Temp disable this before Context is defined
  // TODO re-enable it
  // virtual Status InitRequest(RequestContextBuilder* ctx_builder);

  // Enqueue a func call, thread-safe.
  // func is a stateless function
  virtual common::Status AddTask(int task_type, const void *func,
                                 void **args) = 0;

  // Enqueue through a functor
  // Note, the functor is called immediately.
  inline common::Status
  AddTask(std::function<common::Status()> enqueue_functor) {
    return enqueue_functor();
  }

  // Barrier
  virtual common::Status Sync() = 0;

private:
  const std::string name_;
  WorkQueue(const WorkQueue &) = delete;
  WorkQueue &operator=(const WorkQueue &) = delete;
};

} // namespace brt
