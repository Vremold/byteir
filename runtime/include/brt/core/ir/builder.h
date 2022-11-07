//===- builder.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/ir/ir.h"

#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

// forwarding
namespace brt {
namespace ir {

// forwarding
struct ByREBuilderStructImpl;

class ByREBuilder {
public:
  using TypeAndArgAttrsPack =
      std::tuple<mlir::Type, mlir::byre::EntryFuncArgType, std::string>;

  ByREBuilder();

  ~ByREBuilder();

  mlir::func::FuncOp
  CreateEntryPointFuncSignature(const std::string &func_name,
                                // array of (type, arg type, arg name)
                                const std::vector<TypeAndArgAttrsPack> &types);

  // return a ModuleOp
  mlir::ModuleOp GetModuleOp();

  // return a MLIRContext
  mlir::MLIRContext *GetMLIRContext();

  mlir::Block *GetEntryPointFuncBodyBlock();

  void RecordOperation(mlir::Operation *);

  mlir::Operation *GetRecordOperation();

private:
  std::unique_ptr<ByREBuilderStructImpl> impl_;
};

} // namespace ir
} // namespace brt
