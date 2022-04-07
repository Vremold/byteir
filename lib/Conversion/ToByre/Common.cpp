//===- Common.cpp ---------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//
#include "byteir/Conversion/ToByre/Common.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace llvm;

namespace {

// some code from llvm's AsmPrinter
void appendElementTypeToString(Type type, std::string &out) {
  llvm::TypeSwitch<Type>(type)
      .Case<IndexType>([&](Type) { out += "index"; })
      .Case<BFloat16Type>([&](Type) { out += "bf16"; })
      .Case<Float16Type>([&](Type) { out += "f16"; })
      .Case<Float32Type>([&](Type) { out += "f32"; })
      .Case<Float64Type>([&](Type) { out += "f64"; })
      .Case<Float80Type>([&](Type) { out += "f80"; })
      .Case<Float128Type>([&](Type) { out += "f128"; })
      .Case<IntegerType>([&](IntegerType integerTy) {
        if (integerTy.isSigned()) {
          out += 's';
        } else if (integerTy.isUnsigned()) {
          out += 'u';
        }
        out += 'i' + std::to_string(integerTy.getWidth());
      })
      .Case<ComplexType>([&](ComplexType complexTy) {
        out += 'c';
        appendElementTypeToString(complexTy.getElementType(), out);
      })
      .Case<TupleType>([&](TupleType tupleTy) {
        out += 't';
        for (auto t : tupleTy.getTypes()) {
          out += 'e';
          appendElementTypeToString(t, out);
        }
      })
      .Default([&](Type type) { out += "unknown"; });
}

} // namespace

std::string mlir::getByreKey(StringRef original, TypeRange types,
                             bool appendArgTypes) {

  if (!appendArgTypes)
    return original.str();

  std::string out = original.str();

  for (auto type : types) {
    if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
      Type elementType = memref.getElementType();
      appendElementTypeToString(elementType, out);
    } else {
      out += "unsupport";
    }
  }
  return out;
}
