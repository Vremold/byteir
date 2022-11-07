// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// ===========================================================================
// Modifications Copyright (c) ByteDance.

#pragma once

#include "brt/core/common/logging/sinks/ostream_sink.h"
#include <iostream>

namespace brt {
namespace logging {
/// <summary>
/// A std::clog based ISink
/// </summary>
/// <seealso cref="ISink" />
class CLogSink : public OStreamSink {
public:
  CLogSink() : OStreamSink(std::clog, /*flush*/ true) {}
};
} // namespace logging
} // namespace brt
