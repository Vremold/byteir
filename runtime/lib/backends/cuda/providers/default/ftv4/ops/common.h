#pragma once

#include "brt/core/common/common.h"
#include "fastertransformer_v4/includes/common.h"

namespace brt {
namespace cuda {
namespace ftv4 {
using OperationType = fastertransformerv4::OperationType;
using TransposeType = fastertransformerv4::transposeType;

static inline TransposeType ConvertTransposeType(const std::string &s) {
#define FOR_EACH_TRANS_TYPE(cb)                                                \
  cb(TRANSPOSE0213) cb(TRANSPOSE1203) cb(TRANSPOSE2013)
#define Case(n)                                                                \
  if (s == #n) {                                                               \
    return TransposeType::n;                                                   \
  }
  FOR_EACH_TRANS_TYPE(Case)
#undef Case
#undef FOR_EACH_TRANS_TYPE
  BRT_THROW("unknown transpose type: " + s);
}

} // namespace ftv4
} // namespace cuda
} // namespace brt
