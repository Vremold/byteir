//===- Common.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BYRE_COMMON_H
#define MLIR_DIALECT_BYRE_COMMON_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace byre {

// byre.compute attribute name
inline llvm::StringRef getByreComputeName() { return "byre_compute_name"; }

inline std::string getByrePassThroughArgAttrName() { return "passthrough_arg"; }

inline std::string getByreArgOffsetAttrName() { return "arg_offsets"; }

inline std::string getByreArgRankAttrName() { return "arg_ranks"; }

inline llvm::StringRef getByreForceComputeNameAttrName() {
  return "byre_force_compute_name";
}

inline std::string getByreCallOpReadonlyOperandNumAttrName() {
  return "num_readonly_operand";
}

// byre.compute attributes prefix string
inline std::string getByrePrefix() { return "__byre__"; }

// append attribute with __byre__ prefix string
inline void appendByreComputeAttr(NamedAttrList &attrs, llvm::StringRef name,
                                  Attribute attr) {
  std::string byre_name = getByrePrefix() + name.str();
  attrs.append(byre_name, attr);
}

// return true if the attribute with __byre__ prefix
inline bool isByreComputeAttr(NamedAttribute attr) {
  std::string name = attr.getName().getValue().str();
  return name.find(getByrePrefix()) == 0;
}

// remove __byre__ prefix of the attribute and return
inline NamedAttribute removeByrePrefix(NamedAttribute attr) {
  std::string name =
      attr.getName().getValue().str().substr(getByrePrefix().size());
  StringAttr identifier = StringAttr::get(attr.getName().getContext(), name);
  return NamedAttribute{identifier, attr.getValue()};
}

} // end namespace byre
} // end namespace mlir

#endif // MLIR_DIALECT_BYRE_COMMON_H