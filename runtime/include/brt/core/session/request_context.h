//===- request_context.h --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/common/status.h"
#include "brt/core/framework/event.h"
#include "brt/core/session/session.h"

#include <memory>
#include <string>

namespace brt {

// forward decl
class ExecutionFrame;
class WorkQueue;

/**
 * RequestContext is a class for specific input/output in Session.
 * RequestContext also own ExecutionFrame, and holds WorkQueue and ThreadPool.
 *
 * There are two ways to feed a RequestContext.
 * 1) Bind an existing pointer as an input/output, ownership of which belong to
 * the caller 2) Get a pointer from a given input/output, ownership of which
 * beling to the RequestContext
 */

class RequestContext {
public:
  common::Status BindArg(size_t offset, const void *value);

  void *GetArg(size_t offset);

  // Confirm io binding finished
  void FinishIOBinding();

  common::Status SetShape(size_t offset, const std::vector<int64_t> &shape);

  std::vector<int64_t> GetShape(size_t offset);

  // Synchronize the RequestContext
  common::Status Sync();

  void SetWorkQueue(WorkQueue *wq);

  template <typename T> void AddEventListener(Events::Listener<T> &&listener) {
    events_->AddEventListener<T>(std::move(listener));
  }

  ~RequestContext();

private:
  friend Session;

  /**
   * Private RequestContext constructor
   * Note only Session can construct RequestContext
   * The format of RequestContext is defined throguh Session.
   */
  RequestContext(const Session &session);

  const Session &session_;

  std::unique_ptr<EventListenerManager> events_;

  std::unique_ptr<ExecutionFrame> frame_;

  std::unique_ptr<WorkQueue> wq_;
};

} // namespace brt
