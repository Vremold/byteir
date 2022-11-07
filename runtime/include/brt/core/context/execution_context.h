//===- execution_context.h ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/context/execution_frame.h"

namespace brt {

// Forwarding
class WorkQueue;
class EventListenerManager;
// class ThreadPool;  // TODO add ThreadPool class later

/**
 * ExecutionContext is a light-weight wrapper that has a ExecutionFrame pointer,
 * and other stateful object pointers, such as ThreadPool and WorkQueue. \
 * Note: it doesn't own anything
 */

struct ExecutionContext {
  ExecutionFrame *exec_frame;
  // ThreadPool* thread_pool_;
  WorkQueue *work_queue;
  ExecutionFrame::StateInfo &frame_state_info;
  EventListenerManager *event_listener_manager;

  ExecutionContext(ExecutionFrame *frame, WorkQueue *wq,
                   ExecutionFrame::StateInfo &fs_info,
                   EventListenerManager *event_mgr)
      : exec_frame(frame), work_queue(wq), frame_state_info(fs_info),
        event_listener_manager(event_mgr) {}
};

} // namespace brt
