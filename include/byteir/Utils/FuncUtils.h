//===- FuncUtils.h ------------------------------------------------- C++---===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_FUNCUTILS_H
#define BYTEIR_UTILS_FUNCUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

// get all extra func attrs, filtering out `filterOut`, into `attrs`.
// extra func attrs are attrs not in FuncOp::getAttributeNames()
void getAllExtraFuncAttrs(SmallVectorImpl<mlir::NamedAttribute> &attrs,
                          func::FuncOp func,
                          llvm::ArrayRef<llvm::StringRef> filterOut = {});

// clone all attrs from getAllExtraFuncAttrs of `oldFunc`
// and then add into `newFunc`
void cloneAllExtraFuncAttrs(func::FuncOp oldFunc, func::FuncOp newFunc,
                            llvm::ArrayRef<llvm::StringRef> filterOut = {});

// collapse func region into the first block
void collapseFuncRegion(func::FuncOp func);

// attach compute name, runtime kernel name and trivial arguments offset to
// func
void addGenericFuncAttrs(func::FuncOp func, const std::string &computeName);

} // namespace mlir

#endif // BYTEIR_UTILS_FUNCUTILS_H
