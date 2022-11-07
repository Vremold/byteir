//===- request_context.cc -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/core/session/request_context.h"

#include "brt/core/context/execution_frame.h"
#include "brt/core/context/work_queue.h"

using namespace brt;
using namespace brt::common;

namespace brt {

// TODO move some simple one to header
RequestContext::RequestContext(const Session &session)
    : session_(session), frame_(nullptr), wq_(nullptr),
      events_(std::make_unique<EventListenerManager>()) {}

RequestContext::~RequestContext() {
  if (frame_ && wq_)
    const_cast<Session &>(session_).Cleanup(*this);
}

common::Status RequestContext::BindArg(size_t offset, const void *value) {
  frame_->BindArg(offset, value);
  return Status::OK();
}

void *RequestContext::GetArg(size_t offset) { return frame_->GetArg(offset); }

void RequestContext::FinishIOBinding() { frame_->FinishIOBinding(); }

common::Status RequestContext::SetShape(size_t offset,
                                        const std::vector<int64_t> &shape) {
  return frame_->SetShape(offset, shape);
}

std::vector<int64_t> RequestContext::GetShape(size_t offset) {
  return frame_->GetShape(offset);
}

void RequestContext::SetWorkQueue(WorkQueue *wq) { wq_.reset(wq); }

common::Status RequestContext::Sync() {
  if (wq_ == nullptr) {
    return Status::OK();
  }

  return wq_->Sync();
}

} // namespace brt
