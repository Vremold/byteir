//===- Functils.cpp -------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/FuncUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

void mlir::getAllExtraFuncAttrs(SmallVectorImpl<mlir::NamedAttribute> &attrs,
                                func::FuncOp func,
                                llvm::ArrayRef<llvm::StringRef> filterOut) {
  const auto &defaultFuncAttrs = func::FuncOp::getAttributeNames();

  SmallVector<llvm::StringRef> allFilterOut(defaultFuncAttrs.begin(),
                                            defaultFuncAttrs.end());

  allFilterOut.insert(allFilterOut.end(), filterOut.begin(), filterOut.end());

  auto range =
      llvm::make_filter_range(func->getAttrs(), [&](NamedAttribute attr) {
        return !llvm::is_contained(allFilterOut, attr.getName().getValue());
      });

  attrs.insert(attrs.end(), range.begin(), range.end());
}

void mlir::cloneAllExtraFuncAttrs(func::FuncOp oldFunc, func::FuncOp newFunc,
                                  llvm::ArrayRef<llvm::StringRef> filterOut) {

  SmallVector<mlir::NamedAttribute> attrs;

  getAllExtraFuncAttrs(attrs, oldFunc, filterOut);

  addAttrs(newFunc, attrs);
}

void mlir::collapseFuncRegion(func::FuncOp func) {
  SmallVector<Operation *> ops;
  auto &blocks = func.getBody().getBlocks();
  unsigned tailBlockCnt = 0;

  for (auto it = blocks.begin(); it != blocks.end(); ++it) {
    if (it == blocks.begin())
      continue;

    tailBlockCnt++;
    for (auto &op : *it) {
      ops.push_back(&op);
    }
  }

  for (auto op : ops) {
    op->moveAfter(&func.getBody().front().back());
  }

  for (unsigned i = 0; i < tailBlockCnt; ++i) {
    blocks.back().erase();
  }
}
