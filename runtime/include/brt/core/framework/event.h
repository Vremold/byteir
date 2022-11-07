//===- event.h ------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <functional>
#include <memory>
#include <vector>

namespace brt {
class OpKernelInfo;

struct Events {
  struct BeforeExecutionPlanRun {
    static constexpr uint8_t kIdx = 0;
  };

  struct AfterExecutionPlanRun {
    static constexpr uint8_t kIdx = 1;
  };

  struct BeforeOpKernelRun {
    static constexpr uint8_t kIdx = 2;
    const OpKernelInfo &info;
  };

  struct AfterOpKernelRun {
    static constexpr uint8_t kIdx = 3;
    const OpKernelInfo &info;
  };

  static constexpr uint8_t kSize = 4;

  template <typename T> using Listener = std::function<void(const T &)>;
};

class EventListenerManager {
public:
  template <typename T> void AddEventListener(Events::Listener<T> &&listener) {
    static_assert(T::kIdx < Events::kSize);

    auto sptr = std::make_shared<Events::Listener<T>>(std::move(listener));
    refkeepers.emplace_back(sptr);
    listeners[T::kIdx].emplace_back(sptr);
  }

  template <typename T> void SignalEvent(const T &event) {
    for (auto &&maybe : listeners[T::kIdx]) {
      if (auto listener = maybe.lock()) {
        reinterpret_cast<Events::Listener<T> *>(listener.get())
            ->
            operator()(event);
      }
    }
  }

private:
  std::vector<std::shared_ptr<void>> refkeepers;
  std::array<std::vector<std::weak_ptr<void>>, Events::kSize> listeners;
};

} // namespace brt
