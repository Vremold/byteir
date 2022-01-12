//===- UnrealizedCastToLinalg.cpp -----------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "byteir/Conversion/HloToLinalg/HloToLinalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::linalg;

// some code from mhlo's legalize_to_linalg
namespace {
  SmallVector<Value, 2> ExtractDynamicSizes(OpBuilder& b, Location loc,
    Value tensor,
    Value shape_tensor = nullptr,
    AffineMap permutation = {}) {
    auto tensor_type = tensor.getType().dyn_cast<RankedTensorType>();
    if (!tensor_type) return {};
    SmallVector<Value, 2> dyn_sizes(tensor_type.getRank());
    for (auto& en : llvm::enumerate(tensor_type.getShape())) {
      if (en.value() != ShapedType::kDynamicSize) continue;
      // If a shape tensor is present extract from there.
      if (shape_tensor) {
        Value extract = b.create<tensor::ExtractOp>(
          loc, shape_tensor,
          ValueRange{ b.create<ConstantIndexOp>(loc, en.index()) });
        dyn_sizes[en.index()] =
          b.create<IndexCastOp>(loc, b.getIndexType(), extract);
      }
      else {
        dyn_sizes[en.index()] = b.create<tensor::DimOp>(loc, tensor, en.index());
      }
    }
    if (permutation)
      dyn_sizes = applyPermutationMap(permutation, makeArrayRef(dyn_sizes));
    llvm::erase_value(dyn_sizes, nullptr);  // Strip out placeholders.
    return dyn_sizes;
  }

  Value GetInitTensor(OpBuilder& b, Location loc, ShapedType type,
    ArrayRef<Value> dyn_sizes) {
    return b.create<linalg::InitTensorOp>(loc, dyn_sizes, type.getShape(),
      type.getElementType());
  }

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
  SmallVector<StringRef, 3> GetParallelAndReductionIterators(
    unsigned nLoops, unsigned nReduction) {
    SmallVector<StringRef, 3> res(nLoops - nReduction,
      getParallelIteratorTypeName());
    res.append(nReduction, getReductionIteratorTypeName());
    return res;
  }

  SmallVector<StringRef, 3> GetNParallelLoopsAttrs(unsigned nParallelLoops) {
    return GetParallelAndReductionIterators(nParallelLoops, 0);
  }

  class UnrealizedCastToLinalgConverter : 
    public OpConversionPattern<UnrealizedConversionCastOp> {
  public:
    using OpConversionPattern<UnrealizedConversionCastOp>::
      OpConversionPattern;

    LogicalResult matchAndRewrite(
      UnrealizedConversionCastOp op, 
      UnrealizedConversionCastOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {

      // Find maximum rank / number of loops.
      auto get_rank = [](Value v) {
        return v.getType().cast<ShapedType>().getRank();
      };

      auto is_scalar = [&](Value v) { return get_rank(v) == 0; };
      auto it = llvm::find_if_not(adaptor.getOperands(), is_scalar);
      Value max_rank_arg = adaptor.getOperands().front();
      int64_t nloops = get_rank(max_rank_arg);

      // Find result type, if on tensors.
      ShapedType result_ty = op->getResultTypes().front()
        .template dyn_cast<ShapedType>();

      // Find input/output values and types.
      auto loc = op.getLoc();
      ValueRange inputs = adaptor.getOperands();
      Value output;
  
      auto dyn_sizes = ExtractDynamicSizes(rewriter, loc, max_rank_arg);

      output = GetInitTensor(rewriter, loc, result_ty, dyn_sizes);
    
      // Create indexing maps.
      AffineMap scalar_map = AffineMap::get(nloops, 0, rewriter.getContext());
      AffineMap id_map = rewriter.getMultiDimIdentityMap(nloops);
      SmallVector<AffineMap, 4> maps;
      for (Value v : adaptor.getOperands()) {
        maps.push_back(is_scalar(v) ? scalar_map : id_map);
      }
      maps.push_back(id_map);

      // Build `linalg.generic` op.
      auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc, result_ty , inputs, output, maps,
        GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nested_builder, Location nested_loc, ValueRange args) {
          Type inner_result_ty = getElementTypeOrSelf(output);

          auto inner_cast_op =
            nested_builder.create<UnrealizedConversionCastOp>(loc, inner_result_ty, 
              llvm::to_vector<2>(args.take_front(inputs.size())));
          Value  inner_result = inner_cast_op.getResult(0);

          nested_builder.create<linalg::YieldOp>(loc, inner_result);
        });

      rewriter.replaceOp(op, linalg_op->getResults());
      return success();
    }
  };

  struct UnrealizedCastToLinalgPass
    : public UnrealizedCastToLinalgBase<UnrealizedCastToLinalgPass> {

    UnrealizedCastToLinalgPass() = default;

    void getDependentDialects(DialectRegistry& registry) const final {
      registry.insert<linalg::LinalgDialect, scf::SCFDialect,
        StandardOpsDialect, math::MathDialect, memref::MemRefDialect, 
        shape::ShapeDialect>();
    }

    void runOnFunction() final {
      auto func = getFunction();

      MLIRContext& ctx = getContext();
      OwningRewritePatternList patterns(&ctx);
      ConversionTarget target(ctx);

    target.addLegalDialect<arith::ArithmeticDialect, linalg::LinalgDialect,
                           math::MathDialect, StandardOpsDialect,
                           tensor::TensorDialect, scf::SCFDialect,
                           shape::ShapeDialect>();

      target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&](UnrealizedConversionCastOp op) {
          return !(op.getOperand(0).getType().isa<TensorType>() &&
            op.getResult(0).getType().isa<TensorType>());
        }
        );

      populateUnrealizedCastToLinalgConversionPattern(&ctx, &patterns);
      if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
      }
    }
  };

} // namespace anonymous

void mlir::populateUnrealizedCastToLinalgConversionPattern(
  MLIRContext* context,
  OwningRewritePatternList* patterns) {
  patterns->insert<UnrealizedCastToLinalgConverter>(context);
}

std::unique_ptr<FunctionPass> mlir::createUnrealizedCastToLinalgPass() {
  return std::make_unique<UnrealizedCastToLinalgPass>();
}
