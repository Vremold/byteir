//===- Layout.h ------------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_MEMREF_UTILS_LAYOUT_H
#define BYTEIR_MEMREF_UTILS_LAYOUT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include <functional>

// ByteIR Layout goes through attribute (StringAttr)
// The attribute's key can be gotten through getLayoutAttributeName.
// The attribute's value is defined at either
//   1) Value's DefiningOp's Attribute
//   2) ArgAttr if Value is a BlockArgument (DefiningOp == nullptr)


namespace mlir {
class OpBuilder;
class Value;

llvm::Optional<mlir::AffineMap> createDefaultAffineMap(MLIRContext* ctx, mlir::MemRefType memref);

struct AffineLayoutSpec {
  std::function<llvm::Optional<mlir::AffineMap>(mlir::MLIRContext*, mlir::MemRefType)> createAffineMap;

  AffineLayoutSpec() : createAffineMap([](MLIRContext*, mlir::MemRefType) {return llvm::None; }) {}

  AffineLayoutSpec(std::function<llvm::Optional<mlir::AffineMap>(mlir::MLIRContext*, mlir::MemRefType)> func)
    : createAffineMap(func) {}
};

class AffineLayoutRegistry {
public:
  static AffineLayoutRegistry& getInstance();

  llvm::DenseMap<llvm::StringRef, AffineLayoutSpec> registry;
private:
  AffineLayoutRegistry();
};



inline llvm::StringRef getFuncArgLayoutAttrName() {
  return "layout.name";
}

inline llvm::StringRef getLayoutAttributeName() {
  return "layout";
}

llvm::Optional<llvm::StringRef> getLayoutName(mlir::Value val);

} // namespace mlir

#endif // BYTEIR_MEMREF_UTILS_LAYOUT_H
