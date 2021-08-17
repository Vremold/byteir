//===- EmitUtil.h ---------- ----------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TARGET_EMITUTIL_H
#define BYTEIR_TARGET_EMITUTIL_H

#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"

namespace byteir {

  /// Convenience functions to produce interleaved output with functions returning
  /// a LogicalResult. This is different than those in STLExtras as functions used
  /// on each element doesn't return a string.
  template <typename ForwardIterator, typename UnaryFunctor,
    typename NullaryFunctor>
    inline mlir::LogicalResult
    interleaveWithError(ForwardIterator begin, ForwardIterator end,
      UnaryFunctor eachFn, NullaryFunctor betweenFn) {
    if (begin == end)
      return mlir::success();
    if (mlir::failed(eachFn(*begin)))
      return mlir::failure();
    ++begin;
    for (; begin != end; ++begin) {
      betweenFn();
      if (mlir::failed(eachFn(*begin)))
        return mlir::failure();
    }
    return mlir::success();
  }

  template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
  inline mlir::LogicalResult interleaveWithError(const Container& c,
    UnaryFunctor eachFn,
    NullaryFunctor betweenFn) {
    return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
  }

  template <typename Container, typename UnaryFunctor>
  inline mlir::LogicalResult interleaveCommaWithError(const Container& c,
    llvm::raw_ostream& os,
    UnaryFunctor eachFn) {
    return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
  }

} // namespace byteir

#endif // BYTEIR_TARGET_EMITUTIL_H