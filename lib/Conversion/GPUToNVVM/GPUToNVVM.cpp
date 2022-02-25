//===- CoalescedForToGPU.cpp ----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/GPUToNVVM/GPUToNVVM.h"

#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

#include "../PassDetail.h"
#include <iostream>

using namespace llvm;
using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::LLVM;
using namespace mlir::NVVM;

// Some code from LowerGpuOpsToNVVM
namespace {

template <typename SourceOp>
struct OpToFuncCallLowering : public ConvertOpToLLVMPattern<SourceOp> {
public:
  explicit OpToFuncCallLowering(LLVMTypeConverter& lowering_, StringRef f32Func,
    StringRef f64Func)
    : ConvertOpToLLVMPattern<SourceOp>(lowering_), f32Func(f32Func),
    f64Func(f64Func) {}


  LogicalResult
    matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    using LLVM::LLVMFuncOp;

    static_assert(
      std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
      "expected single result op");

    static_assert(std::is_base_of<OpTrait::SameOperandsAndResultType<SourceOp>,
      SourceOp>::value,
      "expected op with same operand and result types");

    SmallVector<mlir::Value, 1> castedOperands;
    for (mlir::Value operand : adaptor.getOperands())
      castedOperands.push_back(maybeCast(operand, rewriter));

    mlir::Type resultType = castedOperands.front().getType();
    mlir::Type funcType = getFunctionType(resultType, castedOperands);
    StringRef funcName = getFunctionName(
      funcType.cast<LLVM::LLVMFunctionType>().getReturnType());
    if (funcName.empty())
      return failure();

    LLVMFuncOp funcOp = appendOrGetFuncOp(funcName, funcType, op);
    auto callOp = rewriter.create<LLVM::CallOp>(
      op->getLoc(), resultType, SymbolRefAttr::get(funcOp), castedOperands);

    if (resultType == adaptor.getOperands().front().getType()) {
      rewriter.replaceOp(op, { callOp.getResult(0) });
      return success();
    }

    mlir::Value truncated = rewriter.create<LLVM::FPTruncOp>(
      op->getLoc(), adaptor.getOperands().front().getType(),
      callOp.getResult(0));
    rewriter.replaceOp(op, { truncated });
    return success();
  }

private:
  mlir::Value maybeCast(mlir::Value operand, PatternRewriter& rewriter) const {
    mlir::Type type = operand.getType();
    if (!type.isa<Float16Type>())
      return operand;

    return rewriter.create<LLVM::FPExtOp>(
      operand.getLoc(), Float32Type::get(rewriter.getContext()), operand);
  }

  mlir::Type getFunctionType(mlir::Type resultType, ValueRange operands) const {
    SmallVector<mlir::Type> operandTypes(operands.getTypes());
    return LLVM::LLVMFunctionType::get(resultType, operandTypes);
  }

  StringRef getFunctionName(mlir::Type type) const {
    if (type.isa<Float32Type>())
      return f32Func;
    if (type.isa<Float64Type>())
      return f64Func;
    return "";
  }

  LLVM::LLVMFuncOp appendOrGetFuncOp(StringRef funcName, mlir::Type funcType,
    Operation* op) const {
    using LLVM::LLVMFuncOp;

    auto funcAttr = StringAttr::get(op->getContext(), funcName);
    Operation* funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
    if (funcOp)
      return cast<LLVMFuncOp>(*funcOp);

    mlir::OpBuilder b(op->getParentOfType<LLVMFuncOp>());
    return b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  }

  const std::string f32Func;
  const std::string f64Func;
};

// TODO: push back to LLVM later
void populateGpuToNVVMExtConversionPatterns(LLVMTypeConverter& converter,
    RewritePatternSet& patterns) {

  patterns.add<OpToFuncCallLowering<arith::MaxFOp>>(converter, "__nv_fmaxf",
                                                    "__nv_fmax");
  patterns.add<OpToFuncCallLowering<arith::MinFOp>>(converter, "__nv_fminf",
                                                    "__nv_fmin");
}

struct GPUToNVVMExtPass
    : public GPUToNVVMExtBase<GPUToNVVMExtPass> {
  GPUToNVVMExtPass() = default;
  GPUToNVVMExtPass(unsigned indexBitwidth) {
    this->indexBitwidth = indexBitwidth;
  }

  void runOnOperation() override {
    auto m = getOperation();

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(m.getContext(), mlir::DataLayout(m));
    options.emitCWrappers = true;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout) {
      options.overrideIndexBitwidth(indexBitwidth);
    }

    /// MemRef conversion for GPU to NVVM lowering. The GPU dialect uses memory
    /// space 5 for private memory attributions, but NVVM represents private
    /// memory allocations as local `alloca`s in the default address space. This
    /// converter drops the private memory space to support the use case above.
    LLVMTypeConverter converter(m.getContext(), options);
    // Lowering for MMAMatrixType.
    converter.addConversion([&](gpu::MMAMatrixType type) -> mlir::Type {
      return convertMMAToLLVMType(type);
    });

    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    {
      RewritePatternSet patterns(m.getContext());
      populateGpuRewritePatterns(patterns);
      // use void since it will return failure for any gpu.func
      (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
    }


    // llvm lowering
    {
      RewritePatternSet llvmPatterns(m.getContext());
      arith::populateArithmeticToLLVMConversionPatterns(converter,
                                                        llvmPatterns);
      populateGpuToNVVMConversionPatterns(converter, llvmPatterns);
      populateGpuWMMAToNVVMConversionPatterns(converter, llvmPatterns);
      populateMathToLLVMConversionPatterns(converter, llvmPatterns);
      populateMemRefToLLVMConversionPatterns(converter, llvmPatterns);
      populateStdToLLVMConversionPatterns(converter, llvmPatterns);
      populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
      populateStdToLLVMFuncOpConversionPattern(converter, llvmPatterns);
      
      // our extension fixing
      populateGpuToNVVMExtConversionPatterns(converter, llvmPatterns);

      LLVMConversionTarget target(getContext());
      configureGpuToNVVMConversionLegality(target);
      // FIXME: avoid host-call of launchFunc
      target.addLegalOp<gpu::LaunchFuncOp>();

      if (failed(applyPartialConversion(m, target, std::move(llvmPatterns)))) {
        signalPassFailure();
      }
    }
  }



};

} // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createGPUToNVVMExtPass(unsigned indexBitwidth) {
  return std::make_unique<GPUToNVVMExtPass>(indexBitwidth);
}
