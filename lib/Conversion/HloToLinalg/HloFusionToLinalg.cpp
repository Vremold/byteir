//===- HloFusionToLinalg.cpp ----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "byteir/Conversion/HloToLinalg/HloToLinalg.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::mhlo;

namespace {

  // some code from mhlo's legalize_to_linalg
  struct HloFusionToLinalgPass
    : public HloFusionToLinalgBase<HloFusionToLinalgPass> {

    HloFusionToLinalgPass(const std::string& tag)
      : HloFusionToLinalgBase() {
      anchorTag = tag;
    }

    void getDependentDialects(DialectRegistry& registry) const final {
      registry.insert<linalg::LinalgDialect, scf::SCFDialect,
        math::MathDialect, memref::MemRefDialect, shape::ShapeDialect>();
    }

    void runOnOperation() final {
      FuncOp func = getOperation();
      
      bool valid = anchorTag.empty() ||
        func->hasAttrOfType<UnitAttr>(anchorTag);

      // early termination
      if (!valid) return;

      MLIRContext& ctx = getContext();
      OwningRewritePatternList patterns(&ctx);
      ConversionTarget target(ctx);
      target.addLegalDialect<arith::ArithmeticDialect, linalg::LinalgDialect,
                           math::MathDialect, StandardOpsDialect,
                           tensor::TensorDialect, scf::SCFDialect,
                           shape::ShapeDialect>();

      target.addLegalOp<UnrealizedConversionCastOp>();

      mhlo::RemoveSignTypeConverter type_converter;
     
      mhlo::populateHLOToLinalgConversionPattern(&ctx, type_converter, &patterns);

      if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
      }
    }
  };

} // namespace anonymous

std::unique_ptr<OperationPass<FuncOp>>
mlir::createHloFusionToLinalgPass(const std::string &anchorTag) {
  return std::make_unique<HloFusionToLinalgPass>(anchorTag);
}
