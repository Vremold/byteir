//===- ByreDialect.cpp - MLIR Dialect for Runtime implementation -------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//
//
// This file implements the Runtime-related dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Byre/ByreDialect.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include <algorithm>    // for std::any_of

using namespace mlir;
using namespace mlir::byre;

#include "byteir/Dialect/Byre/ByreEnums.cpp.inc"
#include "byteir/Dialect/Byre/ByreOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Common Utilities
//===----------------------------------------------------------------------===//

/// This is a common class used for patterns of the form
/// ```
///    someop(memrefcast(%src)) -> someop(%src)
/// ```
/// It folds the source of the memref.cast into the root operation directly.
static LogicalResult foldMemRefCast(Operation* op) {
  bool folded = false;
  for (OpOperand& operand : op->getOpOperands()) {
    auto castOp = operand.get().getDefiningOp<memref::CastOp>();
    if (castOp && memref::CastOp::canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

// ByreOp has be in FuncOp with EntryPointFunctionAttrName
static LogicalResult verifyOpInEntryPointFunc(Operation* op) {
  auto func = op->getParentOfType<FuncOp>();
  if (!func->hasAttrOfType<UnitAttr>(ByreDialect::getEntryPointFunctionAttrName())) {
    return op->emitError("expected '")
      << ByreDialect::getEntryPointFunctionAttrName() << "' attribute to be attached to '"
      << FuncOp::getOperationName() << "' " << func.getName();
  }
  return success();
}

static bool validEntryFuncArgType(EntryFuncArgType argType) {
  return argType == EntryFuncArgType::Input ||
         argType == EntryFuncArgType::Output ||
         argType == EntryFuncArgType::Weight;
}

//===----------------------------------------------------------------------===//
// ByreDialect
//===----------------------------------------------------------------------===//

void ByreDialect::initialize() {
  addTypes<AsyncTokenType>();
  addOperations<
#define GET_OP_LIST
#include "byteir/Dialect/Byre/ByreOps.cpp.inc"
      >();
}

Type ByreDialect::parseType(DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  // Handle 'async token' types.
  if (keyword == "async.token")
    return AsyncTokenType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown byre type: " + keyword);
  return Type();
}

void ByreDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<AsyncTokenType>([&](Type) { os << "async.token"; })
      .Default([](Type) { llvm_unreachable("unexpected 'byre' type kind"); });
}

LogicalResult ByreDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {

  // ContainerModuleAttr only applied to ModuleOp
  if (attr.getValue().isa<UnitAttr>() &&
      attr.getName().getValue() == getContainerModuleAttrName()) {
    if (!isa<ModuleOp>(op)) {
      return op->emitError("expected '")
             << getContainerModuleAttrName()
             << "' attribute to be attached to '"
             << ModuleOp::getOperationName() << '\'';
    }

    // handle possible ModuleMemorySpaceAttr
    if (auto memSpace =
            op->getAttrOfType<ArrayAttr>(getModuleMemorySpaceAttrName())) {
      // if odd
      if (memSpace.size() & 1) {
        return op->emitError("expected '")
               << getModuleMemorySpaceAttrName() << "' has Even numbers";
      }

      bool isEven = true;
      for (auto it = memSpace.begin(); it != memSpace.end(); ++it) {
        if (isEven && !it->isa<IntegerAttr>()) {
          return op->emitError("expected '") << getModuleMemorySpaceAttrName()
                                             << "'has IntegerAttr in Even";
        }

        if (!isEven && !it->isa<StringAttr>()) {
          return op->emitError("expected '")
                 << getModuleMemorySpaceAttrName() << "'has StringAttr in Odd";
        }
        isEven = !isEven;
      }
    }
  }

  // ModuleMemorySpaceAttr only applied to ModuleOp with ContainerModuleAttr
  if (attr.getValue().isa<ArrayAttr>() &&
      attr.getName().getValue() == getModuleMemorySpaceAttrName()) {
    if (!op->hasAttrOfType<UnitAttr>(getContainerModuleAttrName())) {
      return op->emitError("expected '")
             << getModuleMemorySpaceAttrName()
             << "' attribute to be attached to '"
             << ModuleOp::getOperationName() << "' with '"
             << getContainerModuleAttrName() << '\'';
    }
  }

  // EntryPointFunctionAttr only applied to FuncOp,
  // which under ModuleOp with ContainerModuleAttrName
  if (attr.getValue().isa<UnitAttr>() &&
      attr.getName().getValue() == getEntryPointFunctionAttrName()) {
    if (!isa<FuncOp>(op)) {
      return op->emitError("expected '") << getEntryPointFunctionAttrName()
                                         << "' attribute to be attached to '"
                                         << FuncOp::getOperationName() << '\'';
    }
    auto funcOp = llvm::cast<FuncOp>(op);

    // FuncOp's parent must be ModuleOp with ContainerModuleAttr
    auto parentOp = op->getParentOp();
    if (parentOp == nullptr || !isa<ModuleOp>(parentOp)) {
      return op->emitError("expected '")
             << getEntryPointFunctionAttrName()
             << "' attribute to be attached to '" << FuncOp::getOperationName()
             << "' under '" << ModuleOp::getOperationName() << '\'';
    }

    if (!parentOp->hasAttrOfType<UnitAttr>(getContainerModuleAttrName())) {
      return op->emitError("expected '")
             << getEntryPointFunctionAttrName()
             << "' attribute to be attached to '" << FuncOp::getOperationName()
             << "' under '" << ModuleOp::getOperationName() << "' with '"
             << getContainerModuleAttrName() << '\'';
    }

    // check weights, inputs and outputs
    size_t numInputs = 0, numOutputs = 0, numWeights = 0;
    using ArgType = EntryFuncArgType;
    for (size_t idx = 0; idx < funcOp.getNumArguments(); ++idx) {
      // check argument type
      if (auto argTypeAttr = funcOp.getArgAttrOfType<EntryFuncArgTypeAttr>(
              idx, ByreDialect::getEntryPointFuncArgTypeAttrName())) {
        ArgType argType = argTypeAttr.getValue();
        if (!validEntryFuncArgType(argType)) {
          return op->emitError("invalid argtype '")
                 << stringifyEnum(argType) << "' attached to the argument of '"
                 << FuncOp::getOperationName() << "' under '"
                 << ModuleOp::getOperationName() << '\'';
        }
        if (bitEnumContains(argType, ArgType::Input)) {
          numInputs++;
        }
        if (bitEnumContains(argType, ArgType::Output)) {
          numOutputs++;
        }
        if (bitEnumContains(argType, ArgType::Weight)) {
          numWeights++;
        }
      } else {
        return op->emitError("expected attribute '")
               << getEntryPointFuncArgTypeAttrName()
               << "' to be attached to the argument of '"
               << FuncOp::getOperationName() << "' under '"
               << ModuleOp::getOperationName() << '\'';
      }

      // check argument name
      if (auto argNameAttr = funcOp.getArgAttr(
              idx, ByreDialect::getEntryPointFuncArgNameAttrName())) {
        if (!argNameAttr.isa<StringAttr>()) {
          return op->emitError("expected StringAttr in '")
                 << ByreDialect::getEntryPointFuncArgNameAttrName() << '\'';
        }
      } else {
        return op->emitError("expected attribute '")
               << getEntryPointFuncArgNameAttrName()
               << "' to be attached to the argument of '"
               << FuncOp::getOperationName() << "' under '"
               << ModuleOp::getOperationName() << '\'';
      }
    }

    if (!numOutputs) {
      return op->emitError(
                 "expected at least 1 argument which was attached with '")
             << ByreDialect::getEntryPointFuncArgTypeAttrName()
             << "' attribute contained '"
             << stringifyEnum(EntryFuncArgType::Output) << '\'';
    }

    // FuncOp has no return
    if (funcOp.getNumResults() != 0) {
      return op->emitError("expected no return in ")
             << funcOp.getName() << '\'';
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ComputeOp
//===----------------------------------------------------------------------===/

// verify ComputeOp
static LogicalResult verify(ComputeOp op) {
  return verifyOpInEntryPointFunc(op);
}

FunctionType mlir::byre::ComputeOp::getType() {
  return FunctionType::get(getContext(), getOperandTypes(), {});
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===/

namespace {
  /// Remove copy operations that copy data with the same input and output
  struct EraseIdentityCopyOp : public OpRewritePattern<CopyOp> {
    using OpRewritePattern<CopyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CopyOp copyOp,
      PatternRewriter& rewriter) const override {
      if (copyOp.input() == copyOp.output()) {
        rewriter.eraseOp(copyOp);
        return success();
      }
      return failure();
    }
  };
} // namespace

void CopyOp::getCanonicalizationPatterns(RewritePatternSet& results,
  MLIRContext* context) {
  results.add<EraseIdentityCopyOp>(context);
}

LogicalResult CopyOp::fold(ArrayRef<Attribute>, SmallVectorImpl<OpFoldResult>&) {
  return foldMemRefCast(*this);
}

static LogicalResult verify(CopyOp op) {
  return verifyOpInEntryPointFunc(op);
}

// LWC: ignore Async for now
// 
//===----------------------------------------------------------------------===//
// AsyncOpInterface
//===----------------------------------------------------------------------===//

void byre::addAsyncDependency(Operation *op, Value token) {
  op->insertOperands(0, {token});
  if (!op->template hasTrait<OpTrait::AttrSizedOperandSegments>())
    return;
  auto attrName =
      OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
  auto sizeAttr = op->template getAttrOfType<DenseIntElementsAttr>(attrName);
  if (!sizeAttr)
    return; // Async dependencies is the only variadic operand.
  SmallVector<int32_t, 8> sizes;
  for (auto size : sizeAttr.getValues<APInt>())
    sizes.push_back(size.getSExtValue());
  ++sizes.front();
  op->setAttr(attrName, Builder(op->getContext()).getI32VectorAttr(sizes));
}


#include "byteir/Dialect/Byre/ByreOpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Byre/ByreOps.cpp.inc"
