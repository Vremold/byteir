//===- dtype.h ------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <half/half.hpp>
#include <limits>
#include <type_traits>

#include "brt/core/common/string_view.h"

namespace brt {
enum class DTypeEnum : uint32_t {
  Invalid = 0,
  Float32 = 1,
  Int32 = 2,
  Int64 = 3,
  UInt8 = 4,
  UInt32 = 5,
  Float16 = 6,
  BFloat16 = 7,
  Float64 = 8,
  Bool = 9,
  StringView = 10,
  LastDType,
  Unsupported = LastDType,
};

template <DTypeEnum dtype_enum> struct DTypeTraits;
template <typename ctype> struct ctype_to_dtype;

namespace dtype {
template <typename T, typename SFINAE = void> struct DTypeTraitsImpl;
template <typename T>
struct DTypeTraitsImpl<T,
                       std::enable_if_t<std::numeric_limits<T>::is_specialized>>
    : public std::numeric_limits<T> {
  using impl = std::numeric_limits<T>;

  static constexpr T lower_bound() noexcept {
    return impl::has_infinity ? -impl::infinity() : impl::lowest();
  }

  static constexpr T upper_bound() noexcept {
    return impl::has_infinity ? impl::infinity() : impl::max();
  }
};

template <> struct DTypeTraitsImpl<StringView, void> {};
} // namespace dtype

#define BRT_DEF_DTYPE_TRAITS_FROM_CTYPE(dtype_enum, ctype)                     \
  template <>                                                                  \
  struct DTypeTraits<DTypeEnum::dtype_enum>                                    \
      : public dtype::DTypeTraitsImpl<ctype> {                                 \
    using type_t = ctype;                                                      \
    static_assert(std::is_trivially_copyable<ctype>::value &&                  \
                  std::is_standard_layout<ctype>::value);                      \
  };                                                                           \
  template <> struct ctype_to_dtype<ctype> {                                   \
    static constexpr DTypeEnum value = DTypeEnum::dtype_enum;                  \
  };

BRT_DEF_DTYPE_TRAITS_FROM_CTYPE(Float32, float)
BRT_DEF_DTYPE_TRAITS_FROM_CTYPE(Int32, int32_t)
BRT_DEF_DTYPE_TRAITS_FROM_CTYPE(Int64, int64_t)
BRT_DEF_DTYPE_TRAITS_FROM_CTYPE(UInt8, uint8_t)
BRT_DEF_DTYPE_TRAITS_FROM_CTYPE(UInt32, uint32_t)
BRT_DEF_DTYPE_TRAITS_FROM_CTYPE(Float64, double)
BRT_DEF_DTYPE_TRAITS_FROM_CTYPE(Float16, half_float::half)
BRT_DEF_DTYPE_TRAITS_FROM_CTYPE(Bool, bool)
BRT_DEF_DTYPE_TRAITS_FROM_CTYPE(StringView, StringView)

#undef BRT_DEF_DTYPE_TRAITS_FROM_CTYPE

template <typename ctype>
inline constexpr DTypeEnum dtype_enum_v = ctype_to_dtype<ctype>::value;

} // namespace brt
