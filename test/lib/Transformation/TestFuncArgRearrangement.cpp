//===- TestFuncArgRearrangement.cpp ---------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/FuncArgRearrangement.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace {

// Note these attr names are in unit tests only
constexpr StringRef getByteIRUnitTestAttrName() {
  return "__byteir_unit_test__";
}

constexpr StringRef getFuncArgAttrName() { return "arg"; }

constexpr StringRef getFuncResultAttrName() { return "result"; }

enum RearrangeKind {
  kInit = 0,
  kIdentity = 1,
  kPack = 2,
  kPack2D = 3,
};

struct RearrangeType {
  RearrangeKind kind = RearrangeKind::kInit;
  SmallVector<unsigned> ids;
};

static void createAllIdentity(SmallVector<RearrangeType> &rearrangeTys,
                              unsigned size) {
  for (unsigned i = 0; i < size; ++i) {
    RearrangeType ty;
    ty.kind = RearrangeKind::kIdentity;
    ty.ids.push_back(i);
    rearrangeTys.push_back(ty);
  }
}

static void createRearrange(SmallVector<RearrangeType> &rearrangeTys,
                            DictionaryAttr dict, StringRef key, unsigned size) {
  if (auto attr = dict.get(key)) {

    if (auto arrOfArr = attr.dyn_cast<ArrayAttr>()) {

      for (auto attr : arrOfArr) {
        if (auto arr = attr.dyn_cast<ArrayAttr>()) {
          // at least 2
          if (arr.size() < 2) {
            continue; // skip illegal
          }

          if (auto kindStrAttr = arr[0].dyn_cast<StringAttr>()) {
            RearrangeType ty;
            if (kindStrAttr.str() == "pack") {
              ty.kind = RearrangeKind::kPack;
              for (unsigned i = 1; i < arr.size(); ++i) {
                if (auto intAttr = arr[i].dyn_cast<IntegerAttr>()) {
                  ty.ids.push_back(intAttr.getInt());
                }
              }
            } else if (kindStrAttr.str() == "pack2d") {
              ty.kind = RearrangeKind::kPack2D;
              for (unsigned i = 1; i < arr.size(); ++i) {
                if (auto intAttr = arr[i].dyn_cast<IntegerAttr>()) {
                  ty.ids.push_back(intAttr.getInt());
                }
              }
            } else {
              // identity
              ty.kind = RearrangeKind::kIdentity;
              if (auto intAttr = arr[1].dyn_cast<IntegerAttr>()) {
                ty.ids.push_back(intAttr.getInt());
              } else {
                ty.ids.push_back(0); // fallback
              }
            }
            rearrangeTys.push_back(ty);
          } // has strAttr
        }   // is ArrayAttr
      }     // for arrOfArr

      return;
    }
  }

  createAllIdentity(rearrangeTys, size);
  return;
}

// check whether RearrangeType is valid
// and compute reverse RearrangeType
static bool checkAndComputeReverse(SmallVector<RearrangeType> &reverseTys,
                                   ArrayRef<RearrangeType> rearrangeTys,
                                   unsigned size) {
  reverseTys.resize(size);
  DenseSet<unsigned> visited;

  for (unsigned i = 0; i < rearrangeTys.size(); ++i) {
    const RearrangeType &r = rearrangeTys[i];
    for (auto id : r.ids) {
      if (visited.contains(id))
        return false;

      visited.insert(id);

      reverseTys[id].kind = r.kind;
      reverseTys[id].ids.push_back(i);
    }
  }
  return true;
}

// compuate concat along last axis
// Note: cannot directly use mhlo::ConcatenateOp::inferReturnTypes, since no
// Value
static Type packLastInferType(ArrayRef<Type> types) {
  auto firstTy = types.front();

  if (auto firstTensorTy = firstTy.dyn_cast<TensorType>()) {

    auto firstShape = firstTensorTy.getShape();

    SmallVector<int64_t> packShape(firstShape.begin(), firstShape.end());

    if (packShape.back() == ShapedType::kDynamicSize) {
      return firstTy;
    }

    for (unsigned i = 1; i < types.size(); ++i) {
      if (auto tensorTy = types[i].dyn_cast<TensorType>()) {
        auto curShape = tensorTy.getShape();

        // check every shape except last
        if (firstShape.drop_back() != curShape.drop_back()) {
          return Type();
        }

        // accumulate last
        if (curShape.back() != ShapedType::kDynamicSize) {
          packShape.back() += curShape.back();
        } else {
          packShape.back() = ShapedType::kDynamicSize;
        }
      } else {
        return Type();
      }
    }

    return firstTensorTy.clone(packShape, firstTensorTy.getElementType());
  }

  return Type();
}

static TensorType reshape2DLast(TensorType tensorTy) {
  auto shape = tensorTy.getShape();
  SmallVector<int64_t> retShape;
  retShape.push_back(shape.front());
  int64_t last = 1;

  for (size_t i = 1; i < shape.size(); ++i) {
    if (shape[i] == ShapedType::kDynamicSize) {
      last = ShapedType::kDynamicSize;
      break;
    }
    last *= shape[i];
  }
  retShape.push_back(last);
  return tensorTy.clone(retShape, tensorTy.getElementType());
}

// compuate concat(reshape, axis == last)
static Type reshapeAndPackLastInferType(ArrayRef<Type> types) {
  auto firstTy = types.front();

  if (auto firstTensorTy = firstTy.dyn_cast<TensorType>()) {

    auto reshapedFirstTy = reshape2DLast(firstTensorTy);
    if (reshapedFirstTy.getShape().back() == ShapedType::kDynamicSize) {
      return reshapedFirstTy;
    }

    SmallVector<int64_t> packShape(reshapedFirstTy.getShape().begin(),
                                   reshapedFirstTy.getShape().end());

    for (unsigned i = 1; i < types.size(); ++i) {
      if (auto tensorTy = types[i].dyn_cast<TensorType>()) {
        auto reshapedCurTy = reshape2DLast(tensorTy);
        auto reshapedCurShape = reshapedCurTy.getShape();

        // check first dim
        if (packShape[0] != reshapedCurShape[0]) {
          return Type();
        }

        if (reshapedCurShape.back() == ShapedType::kDynamicSize) {
          return reshapedFirstTy;
        } else {
          packShape[1] += reshapedCurShape[1];
        }

      } else {
        return Type();
      }
    }

    return reshapedFirstTy.clone(packShape, reshapedFirstTy.getElementType());
  }

  return Type();
}

static Type createNewValueType(MLIRContext *ctx, const RearrangeType &r,
                               ArrayRef<Type> types) {
  if (r.kind == RearrangeKind::kIdentity) {
    return types[r.ids.back()];
  } else {
    // RearrangeKind::kPack or kPack2D
    SmallVector<Type> filteredTy;
    for (auto id : r.ids) {
      filteredTy.push_back(types[id]);
    }

    if (r.kind == RearrangeKind::kPack) {
      return packLastInferType(filteredTy);
    } else {
      // r.kind == RearrangeKind::kPack2D
      return reshapeAndPackLastInferType(filteredTy);
    }
  }
  return Type();
}

static FunctionType
createNewFuncType(func::FuncOp func, ArrayRef<RearrangeType> rearrangeArgs,
                  ArrayRef<RearrangeType> rearrangeResults) {
  // handle args
  SmallVector<Type> argTys;
  for (const auto &r : rearrangeArgs) {
    argTys.push_back(createNewValueType(func.getContext(), r,
                                        func.getFunctionType().getInputs()));
  }

  // handle results
  SmallVector<Type> retTys;

  for (const auto &r : rearrangeResults) {
    retTys.push_back(createNewValueType(func.getContext(), r,
                                        func.getFunctionType().getResults()));
  }

  return FunctionType::get(func.getContext(), argTys, retTys);
}

static Value packArgs(OpBuilder &b, ArrayRef<unsigned> ids,
                      ArrayRef<Value> values) {
  SmallVector<Value> args;
  for (auto id : ids) {
    args.push_back(values[id]);
  }

  if (auto tensorTy = args.back().getType().dyn_cast<TensorType>()) {
    auto rank = tensorTy.getRank();
    // only support last dim for now
    auto concat = b.create<mhlo::ConcatenateOp>(UnknownLoc::get(b.getContext()),
                                                args, rank - 1);
    return concat.getResult();
  }

  return Value();
}

static Value reshapeAndPack2DArgs(OpBuilder &b, ArrayRef<unsigned> ids,
                                  ArrayRef<Value> values) {
  SmallVector<Value> args;
  for (auto id : ids) {
    auto arg = values[id];
    if (!arg.getType().isa<TensorType>()) {
      return Value();
    }
    args.push_back(values[id]);
  }

  SmallVector<Value> reshapedArgs;
  for (auto arg : args) {
    auto argTensorTy = arg.getType().cast<TensorType>();
    auto reshapedTensorTy = reshape2DLast(argTensorTy);
    auto reshape = b.create<mhlo::ReshapeOp>(UnknownLoc::get(b.getContext()),
                                             reshapedTensorTy, arg);
    reshapedArgs.push_back(reshape.getResult());
  }

  auto concat = b.create<mhlo::ConcatenateOp>(UnknownLoc::get(b.getContext()),
                                              reshapedArgs, 1);
  return concat.getResult();
}

static void setFixedDimOfShape(SmallVector<int64_t> &begins,
                               SmallVector<int64_t> &ends, Type ty,
                               bool hasReshape) {
  if (auto tensorTy = ty.dyn_cast<TensorType>()) {
    auto targetTy = hasReshape ? reshape2DLast(tensorTy) : tensorTy;
    for (unsigned i = 0; i < ends.size() - 1; ++i) {
      begins[i] = 0;
      ends[i] = targetTy.getShape()[i];
    }
  }
}

static void accumulateDimOfStaticShape(SmallVector<int64_t> &ends, Type ty,
                                       bool hasReshape) {
  if (auto tensorTy = ty.dyn_cast<TensorType>()) {
    auto targetTy = hasReshape ? reshape2DLast(tensorTy) : tensorTy;
    ends.back() += targetTy.getShape().back();
  }
}

static void computeBeginAndEndForUnPack(SmallVector<int64_t> &begins,
                                        SmallVector<int64_t> &ends,
                                        unsigned rank, unsigned id,
                                        ArrayRef<unsigned> ids,
                                        ArrayRef<Type> types, bool hasReshape) {
  begins.resize(rank, 0);
  ends.resize(rank, 0);

  for (auto i : ids) {
    begins = ends; // begins as previous ends
    accumulateDimOfStaticShape(ends, types[i], hasReshape);
    if (id == i) {
      break;
    }
  }
  setFixedDimOfShape(begins, ends, types[ids.back()], hasReshape);
}

static Value unPackArg(OpBuilder &b, ArrayRef<int64_t> begins,
                       ArrayRef<int64_t> ends, ArrayRef<int64_t> strides,
                       Value val) {
  if (auto tesorTy = val.getType().dyn_cast<TensorType>()) {
    auto indicesTy = RankedTensorType::get(tesorTy.getRank(), b.getI64Type());
    auto slice =
        b.create<mhlo::SliceOp>(UnknownLoc::get(b.getContext()), val,
                                DenseIntElementsAttr::get(indicesTy, begins),
                                DenseIntElementsAttr::get(indicesTy, ends),
                                DenseIntElementsAttr::get(indicesTy, strides));
    return slice.getResult();
  }

  return Value();
}

static Value reshapeArg(OpBuilder &b, Type type, Value val) {
  auto reshape =
      b.create<mhlo::ReshapeOp>(UnknownLoc::get(b.getContext()), type, val);
  return reshape.getResult();
}

// this test reads the rearranger from attr
// The attr is an Array of Array
class FuncArgRearrangerTest : public FuncArgRearrangerBase {
public:
  FuncArgRearrangerTest(func::FuncOp f) : FuncArgRearrangerBase(), funcOp(f) {}

  bool init() override {
    // early termination
    if (!funcOp->hasAttrOfType<DictionaryAttr>(getByteIRUnitTestAttrName()))
      return false;

    auto dict =
        funcOp->getAttrOfType<DictionaryAttr>(getByteIRUnitTestAttrName());

    // handle args
    createRearrange(rearrangeArgs, dict, getFuncArgAttrName(),
                    funcOp.getNumArguments());

    // handle results
    createRearrange(rearrangeResults, dict, getFuncResultAttrName(),
                    funcOp.getNumResults());

    // compute reverse here
    if (!checkAndComputeReverse(reverseArgs, rearrangeArgs,
                                funcOp.getNumArguments())) {
      return false;
    }

    if (!checkAndComputeReverse(reverseResults, rearrangeResults,
                                funcOp.getNumResults())) {
      return false;
    }

    // compute newFuncType here
    newFuncType = createNewFuncType(funcOp, rearrangeArgs, rearrangeResults);
    return true;
  }

  FunctionType getFunctionType() override { return newFuncType; }

  Value getOrCreateNewFromOldFuncArg(OpBuilder &b, unsigned newId,
                                     ArrayRef<Value> oldValues) override {
    auto r = rearrangeArgs[newId];

    if (r.kind == RearrangeKind::kIdentity) {
      return oldValues[r.ids.back()];
    } else if (r.kind == RearrangeKind::kPack) {
      // RearrangeKind::kPack
      return packArgs(b, r.ids, oldValues);
    } else {
      // RearrangeKind::kPack2D
      return reshapeAndPack2DArgs(b, r.ids, oldValues);
    }

    return Value();
  }

  Optional<Value>
  getOrCreateOldFromNewFuncArg(OpBuilder &b, unsigned oldId,
                               ArrayRef<Value> newValues) override {
    auto r = reverseArgs[oldId];

    if (r.kind == RearrangeKind::kIdentity) {
      return newValues[r.ids.back()];
    } else {
      // RearrangeKind::kPack or kPack2D
      auto rForward = rearrangeArgs[r.ids.back()];
      SmallVector<int64_t> begins;
      SmallVector<int64_t> ends;

      auto newVal = newValues[r.ids.back()];

      if (auto newTensorTy = newVal.getType().dyn_cast<TensorType>()) {
        auto rank = newTensorTy.getRank();
        bool hasReshape = r.kind == RearrangeKind::kPack2D;

        computeBeginAndEndForUnPack(begins, ends, rank, oldId, rForward.ids,
                                    funcOp.getFunctionType().getInputs(),
                                    hasReshape);

        SmallVector<int64_t> strides(rank, 1);
        auto unPackVal = unPackArg(b, begins, ends, strides, newVal);

        if (hasReshape) {
          // if RearrangeKind::kPack2D
          auto oldTy = funcOp.getFunctionType().getInput(oldId);
          return reshapeArg(b, oldTy, unPackVal);
        }

        // if RearrangeKind::kPack
        return unPackVal;
      }
    }

    return llvm::None;
  }

  Value getOrCreateNewFromOldFuncResult(OpBuilder &b, unsigned newId,
                                        ArrayRef<Value> oldValues) override {
    auto r = rearrangeResults[newId];

    if (r.kind == RearrangeKind::kIdentity) {
      return oldValues[r.ids.back()];
    } else if (r.kind == RearrangeKind::kPack) {
      // RearrangeKind::kPack
      return packArgs(b, r.ids, oldValues);
    } else {
      // RearrangeKind::kPack2D
      return reshapeAndPack2DArgs(b, r.ids, oldValues);
    }

    return Value();
  }

  Optional<Value>
  getOrCreateOldFromNewFuncResult(OpBuilder &b, unsigned oldId,
                                  ArrayRef<Value> newValues) override {
    auto r = reverseResults[oldId];

    if (r.kind == RearrangeKind::kIdentity) {
      return newValues[r.ids.back()];
    } else {
      // RearrangeKind::kPack or kPack2D
      auto rForward = rearrangeResults[r.ids.back()];
      SmallVector<int64_t> begins;
      SmallVector<int64_t> ends;

      auto newVal = newValues[r.ids.back()];

      if (auto newTensorTy = newVal.getType().dyn_cast<TensorType>()) {
        auto rank = newTensorTy.getRank();
        bool hasReshape = r.kind == RearrangeKind::kPack2D;
        computeBeginAndEndForUnPack(begins, ends, rank, oldId, rForward.ids,
                                    funcOp.getFunctionType().getResults(),
                                    hasReshape);

        SmallVector<int64_t> strides(rank, 1);
        auto unPackVal = unPackArg(b, begins, ends, strides, newVal);

        if (hasReshape) {
          // if r.kind == RearrangeKind::kPack2D
          auto oldTy = funcOp.getFunctionType().getResult(oldId);
          return reshapeArg(b, oldTy, unPackVal);
        }
        // if r.kind == RearrangeKind::kPack
        return unPackVal;
      }
    }

    return llvm::None;
  }

private:
  // local meta data
  func::FuncOp funcOp;

  FunctionType newFuncType;

  SmallVector<RearrangeType> rearrangeArgs;
  SmallVector<RearrangeType> rearrangeResults;

  SmallVector<RearrangeType> reverseArgs;
  SmallVector<RearrangeType> reverseResults;
};

class FuncArgRearrangerBuilderTest : public FuncArgRearrangerBuilderBase {
public:
  std::unique_ptr<FuncArgRearrangerBase>
  createFuncArgRearranger(func::FuncOp f) override {
    return std::make_unique<FuncArgRearrangerTest>(f);
  }
};

struct TestFuncArgRearrangementPass
    : public PassWrapper<TestFuncArgRearrangementPass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFuncArgRearrangementPass)

  StringRef getArgument() const final { return "test-rearrange-func-arg"; }

  StringRef getDescription() const final {
    return "Test Func Arg Rearrangement";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
  }

  void runOnOperation() override {
    auto m = getOperation();

    std::unique_ptr<FuncArgRearrangerBuilderBase> testBuilder =
        std::make_unique<FuncArgRearrangerBuilderTest>();

    auto testAttr = getByteIRUnitTestAttrName();

    OpPassManager pm(m.getOperationName());
    pm.addPass(
        createFuncArgRearrangementPass(testBuilder.get(), testAttr.str()));
    addCleanUpPassPipeline(pm);
    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace byteir {
namespace test {
void registerTestFuncArgRearrangementPass() {
  PassRegistration<TestFuncArgRearrangementPass>();
}
} // namespace test
} // namespace byteir