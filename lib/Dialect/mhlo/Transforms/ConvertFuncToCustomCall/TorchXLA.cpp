//===- TorchXLA.cpp ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/ConvertFuncToCustomCall/TorchXLA.h"
#include "byteir/Dialect/mhlo/Transforms/ConvertFuncToCustomCall.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringRef.h"

#include "../PassDetail.h"

// tentative
// FIXME: remove the entire file and folder after TorchXLA pipeline settled down

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

NamedAttrList getDefaultAttrs(MLIRContext *ctx, StringRef callTargetName,
                              bool hasSideEffect) {
  NamedAttrList attrs;
  attrs.append(::llvm::StringRef("call_target_name"),
               StringAttr::get(ctx, callTargetName));

  attrs.append(::llvm::StringRef("has_side_effect"),
               BoolAttr::get(ctx, hasSideEffect));

  // the rest ones use default values
  attrs.append(::llvm::StringRef("backend_config"), StringAttr::get(ctx));

  attrs.append(::llvm::StringRef("api_version"),
               CustomCallApiVersionAttr::get(
                   ctx, CustomCallApiVersion::API_VERSION_ORIGINAL));

  attrs.append(::llvm::StringRef("called_computations"),
               ArrayAttr::get(ctx, {}));

  return attrs;
}

// get real number of type with TupleType in mind
unsigned getRealNumTypes(ArrayRef<Type> types) {
  unsigned num = 0;
  for (Type type : types) {
    auto tupleType = type.dyn_cast<TupleType>();
    if (tupleType == nullptr) {
      num++;
      continue;
    }

    num += getRealNumTypes(tupleType.getTypes());
  }
  return num;
}

void convertTorchMax(func::FuncOp func, ModuleOp m) {
  auto maybeSymbolUses = func.getSymbolUses(m);
  MLIRContext *ctx = m.getContext();
  OpBuilder b(ctx);
  DenseIntElementsAttr axisAttr;
  func.walk([&](mhlo::ReduceOp reduceOp) {
    axisAttr = reduceOp.getDimensions();
    return WalkResult::interrupt();
  });
  int64_t axisInt = (*axisAttr.begin()).getSExtValue();
  unsigned numRes = getRealNumTypes(func.getFunctionType().getResults());

  for (SymbolTable::SymbolUse symbolUse : *maybeSymbolUses) {
    if (auto callOp = dyn_cast<func::CallOp>(symbolUse.getUser())) {
      auto loc = callOp.getLoc();
      b.setInsertionPoint(callOp);
      auto operands = callOp.getOperands();

      // Both the result types of each case are the same as the func op's. Tuple
      // type isn't flattened here.
      if (numRes == 1) {
        // it is acutally a reduce_max
        NamedAttrList reduceMaxAttrs =
            getDefaultAttrs(ctx, getReduceMaxName(), false);
        llvm::SmallVector<NamedAttribute> extReduceMaxAttrArray;
        extReduceMaxAttrArray.push_back(
            b.getNamedAttr(StringRef("axis"), axisAttr));
        extReduceMaxAttrArray.push_back(
            b.getNamedAttr(StringRef("keep_dims"), BoolAttr::get(ctx, false)));
        DictionaryAttr extReduceMaxAttr =
            b.getDictionaryAttr(extReduceMaxAttrArray);
        reduceMaxAttrs.append(getByteIRCustomCallAttrName(), extReduceMaxAttr);
        auto reduceMaxOp = b.create<mhlo::CustomCallOp>(
            loc, callOp.getResultTypes(), operands, reduceMaxAttrs);
        callOp.getResult(0).replaceAllUsesWith(reduceMaxOp->getResult(0));
      } else {
        // it is acutally a arg_max with two results
        NamedAttrList argMaxAttrs =
            getDefaultAttrs(ctx, getArgMaxName(), false);
        llvm::SmallVector<NamedAttribute> extArgMaxAttrArray;
        extArgMaxAttrArray.push_back(
            b.getNamedAttr(StringRef("axis"), b.getI64IntegerAttr(axisInt)));
        extArgMaxAttrArray.push_back(
            b.getNamedAttr(StringRef("keep_dims"), BoolAttr::get(ctx, false)));
        extArgMaxAttrArray.push_back(b.getNamedAttr(
            StringRef("select_last_index"), BoolAttr::get(ctx, false)));
        DictionaryAttr extArgMaxAttr = b.getDictionaryAttr(extArgMaxAttrArray);
        argMaxAttrs.append(getByteIRCustomCallAttrName(), extArgMaxAttr);
        auto argMaxOp = b.create<mhlo::CustomCallOp>(
            loc, callOp.getResultTypes(), operands, argMaxAttrs);
        for (auto it : llvm::zip(callOp.getResults(), argMaxOp->getResults())) {
          Value oldResult = std::get<0>(it);
          Value newResult = std::get<1>(it);
          oldResult.replaceAllUsesWith(newResult);
        }
      }

      // erase call op
      callOp->erase();
    }
  }
}

struct FuncToCustomCallConverterTorchXLA
    : public FuncToCustomCallConverterLookup {

  FuncToCustomCallConverterTorchXLA(
      const llvm::StringMap<std::string> &externalMap,
      const llvm::StringMap<std::function<void(func::FuncOp, ModuleOp m)>>
          &funcNameToCustomizedCvs)
      : FuncToCustomCallConverterLookup() {
    for (const auto &it : externalMap) {
      funcNameToCustomMeta.try_emplace(it.first(), it.second, false);
    }
    funcNameToCustomizedConversion = funcNameToCustomizedCvs;
  }
  virtual ~FuncToCustomCallConverterTorchXLA() {}
};

struct ConvertFuncToCustomCallTorchXLAPass
    : public ConvertFuncToCustomCallTorchXLABase<
          ConvertFuncToCustomCallTorchXLAPass> {

  void runOnOperation() override {
    auto m = getOperation();
    llvm::StringMap<std::string> funcNameToCallTarget;
    funcNameToCallTarget.try_emplace("aten.erf", getErfName());

    llvm::StringMap<std::function<void(func::FuncOp, ModuleOp m)>>
        funcNameToCustomizedConversion;
    funcNameToCustomizedConversion.try_emplace("aten.max", convertTorchMax);

    std::unique_ptr<FuncToCustomCallConverterBase> converter =
        std::make_unique<FuncToCustomCallConverterTorchXLA>(
            funcNameToCallTarget, funcNameToCustomizedConversion);

    OpPassManager pm(m.getOperationName());
    pm.addPass(createConvertFuncToCustomCallPass(converter.get()));
    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertFuncToCustomCallTorchXLAPass() {
  return std::make_unique<ConvertFuncToCustomCallTorchXLAPass>();
}
