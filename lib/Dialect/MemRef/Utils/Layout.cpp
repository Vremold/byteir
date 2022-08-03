//===- Layout.cpp
//----------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/MemRef/Utils/Layout.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Value.h"

using namespace mlir;

namespace {
llvm::Optional<mlir::AffineMap> createTestAffineMap(MLIRContext *ctx,
                                                    mlir::MemRefType memref) {
  if (memref.getRank() == 2) {
    AffineExpr x0 = mlir::getAffineDimExpr(0, ctx);
    AffineExpr x1 = mlir::getAffineDimExpr(1, ctx);
    SmallVector<AffineExpr, 2> results;
    results.push_back(x1);
    results.push_back(x0);
    return AffineMap::get(2, 0, results, ctx);
  }

  return llvm::None;
}
} // namespace

llvm::Optional<mlir::AffineMap>
mlir::createDefaultAffineMap(MLIRContext *ctx, mlir::MemRefType memref) {
  return AffineMap::get(memref.getRank(), memref.getNumDynamicDims(), ctx);
}

AffineLayoutRegistry::AffineLayoutRegistry() {
  // insert a test_layout for test purpose
  AffineLayoutSpec spec(createTestAffineMap);
  registry.try_emplace("test_affine_layout", spec);
}

AffineLayoutRegistry &mlir::AffineLayoutRegistry::getInstance() {
  static AffineLayoutRegistry instance;
  return instance;
}

llvm::Optional<llvm::StringRef> mlir::getLayoutName(mlir::Value val) {

  if (auto defOp = val.getDefiningOp()) {
    if (defOp->hasAttrOfType<StringAttr>(getLayoutAttributeName())) {
      return defOp->getAttrOfType<StringAttr>(getLayoutAttributeName())
          .getValue();
    }
  } else if (auto arg = val.dyn_cast<BlockArgument>()) {
    Region *region = arg.getParentRegion();
    if (region == nullptr)
      return llvm::None;

    if (auto funcOp = region->getParentOfType<func::FuncOp>()) {
      if (auto argAttr = funcOp.getArgAttrOfType<StringAttr>(
              arg.getArgNumber(), getFuncArgLayoutAttrName())) {
        return argAttr.getValue();
      }
    }
  }

  return None;
}
