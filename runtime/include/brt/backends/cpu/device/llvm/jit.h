//===- module.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <ostream>
#include <string>

#include "brt/core/common/common.h"

namespace brt {
namespace cpu {

class LLVMJITImpl;

class LLVMJIT {
public:
  LLVMJIT();
  ~LLVMJIT();

  static LLVMJIT *Instance();

  common::Status LoadFromFile(const std::string &path);

  // \p buf should be a pointer to llvm ThreadSafeModule
  common::Status LoadFromBuffer(void *buf);

  // return whether the \p symbol_name is found
  // if \p symbol_name is found address of the corresponding symbol would be
  // set to \p symbol
  common::Status Lookup(const std::string &symbol_name, void **symbol);
  common::Status LookupPacked(const std::string &symbol_name, void **symbol);

  common::Status RegisterSymbol(const std::string &symbol_name, void *symbol);

  common::Status PrintOptimizedModule(const std::string &indentifier,
                                      std::ostream &os);
  common::Status DumpObject(const std::string &indentifier, std::ostream &os);

private:
  std::unique_ptr<LLVMJITImpl> impl;
};

} // namespace cpu
} // namespace brt
