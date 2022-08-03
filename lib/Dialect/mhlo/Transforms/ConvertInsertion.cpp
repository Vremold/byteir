//===- ConvertInsertion.cpp -----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/ConvertInsertion.h"
#include "PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::func;
using namespace mlir::mhlo;

namespace {

class ConvertInsertionPass : public ::mlir::OperationPass<ModuleOp> {
public:
  using Base = ConvertInsertionPass;

  ConvertInsertionPass()
      : ::mlir::OperationPass<ModuleOp>(
            ::mlir::TypeID::get<ConvertInsertionPass>()) {}

  ConvertInsertionPass(const ConvertInsertionPass &other)
      : ::mlir::OperationPass<ModuleOp>(other) {}

  explicit ConvertInsertionPass(ConvertRuleBase *externalCollector)
      : ::mlir::OperationPass<ModuleOp>(
            ::mlir::TypeID::get<ConvertInsertionPass>()),
        collector(externalCollector) {}

// Note command-line was disable in this pass, due to it using a class to drive
// Please use TestConvertInsertion (test-insert-convert) in command-line
#if 0 
  /// Returns the command-line argument attached to this pass.
   static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("insert-convert");
  }
  ::llvm::StringRef getArgument() const override { return "insert-convert"; }

  ::llvm::StringRef getDescription() const override {
    return "Insert ConvertOps (for CallOps and FuncOps now)";
  }
#endif

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ConvertInsertion");
  }
  ::llvm::StringRef getName() const override { return "ConvertInsertion"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<ConvertInsertionPass>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<ConvertInsertionPass>(
        *static_cast<const ConvertInsertionPass *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {

    registry.insert<mlir::mhlo::MhloDialect>();
  }

  void runOnOperation() override;

protected:
  ConvertRuleBase *collector = nullptr;
};

} // namespace

void ConvertInsertionPass::runOnOperation() {
  auto m = getOperation();
  auto ctx = m.getContext();

  // early return if no collector
  if (nullptr == collector) {
    return;
  }

  // iterte all func
  for (auto func : m.getOps<func::FuncOp>()) {
    if (!collector->checkFunc(func)) {
      continue;
    }

    FunctionType funcType = func.getFunctionType();
    mlir::SmallVector<Type> argTypes;
    argTypes.reserve(funcType.getNumInputs());
    mlir::SmallVector<Type> retTypes;
    retTypes.reserve(funcType.getNumResults());

    // arg
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      auto oldTy = funcType.getInput(i);
      auto maybeTensor = collector->checkType(oldTy);
      if (maybeTensor.hasValue()) {
        argTypes.push_back(maybeTensor.getValue());
      } else {
        argTypes.push_back(oldTy);
      }
    }

    // results
    for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
      auto oldTy = funcType.getResult(i);
      auto maybeTensor = collector->checkType(oldTy);
      if (maybeTensor.hasValue()) {
        retTypes.push_back(maybeTensor.getValue());
      } else {
        retTypes.push_back(oldTy);
      }
    }

    // insert convert for a call
    auto maybeSymbolUses = func.getSymbolUses(m);
    OpBuilder b(ctx);
    for (SymbolTable::SymbolUse symbolUse : *maybeSymbolUses) {
      if (auto callOp = dyn_cast<CallOp>(symbolUse.getUser())) {
        auto loc = callOp.getLoc();
        // insert before callOp for arg convert
        b.setInsertionPoint(callOp);
        for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
          auto arg = callOp.getOperand(i);
          auto convertOp = b.create<mhlo::ConvertOp>(loc, argTypes[i], arg);
          callOp.setOperand(i, convertOp.getResult());
        }

        // insert after callOp for return convert
        b.setInsertionPointAfter(callOp);
        for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
          auto ret = callOp.getResult(i);
          ret.setType(retTypes[i]);
          auto convertOp =
              b.create<mhlo::ConvertOp>(loc, funcType.getResult(i), ret);
          ret.replaceAllUsesExcept(convertOp.getResult(), convertOp);
        }
      }
    }

    // rewrite FunctionType
    func.setType(FunctionType::get(ctx, argTypes, retTypes));

  } // end iterate func
}

mlir::ConvertOnlyCheckElementType::ConvertOnlyCheckElementType(
    mlir::StringRef strRrf)
    : anchorAttr(strRrf.str()) {}

bool mlir::ConvertOnlyCheckElementType::checkFunc(func::FuncOp func) {
  return func->hasAttr(anchorAttr);
}

llvm::Optional<mlir::TensorType>
mlir::ConvertOnlyCheckElementType::checkType(mlir::Type type) {
  if (auto TensorTy = type.dyn_cast<TensorType>()) {
    auto elementTy = TensorTy.getElementType();
    if (convertElementType.count(elementTy) > 0) {
      return TensorTy.clone(convertElementType[elementTy]);
    }
  }
  return llvm::None;
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertInsertionPass(ConvertRuleBase *collector) {
  return std::make_unique<ConvertInsertionPass>(collector);
}
