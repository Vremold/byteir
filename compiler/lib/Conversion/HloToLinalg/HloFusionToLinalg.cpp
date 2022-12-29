//===- HloFusionToLinalg.cpp ----------------------------------*--- C++ -*-===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
// Some code comes from legalize_to_linalg.cc in TensorFlow
// Original license:
//
// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/HloToLinalg/HloToLinalg.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::mhlo;

namespace {

struct HloFusionToLinalgPass
    : public HloFusionToLinalgBase<HloFusionToLinalgPass> {

  HloFusionToLinalgPass(StringRef tag, bool enablePrimitiveOps)
      : HloFusionToLinalgBase() {
    anchorTag = tag.str();
    this->enablePrimitiveOps = enablePrimitiveOps;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, math::MathDialect,
                    memref::MemRefDialect, shape::ShapeDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();

    bool valid = anchorTag.empty() || func->hasAttrOfType<UnitAttr>(anchorTag);

    // early termination
    if (!valid)
      return;

    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                           func::FuncDialect, linalg::LinalgDialect,
                           math::MathDialect, tensor::TensorDialect,
                           scf::SCFDialect, shape::ShapeDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    auto typeConverter = createHloToLinalgTypeConverter();

    mhlo::populateHloToLinalgConversionPattern(&ctx, *typeConverter, &patterns,
                                               enablePrimitiveOps);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPartialConversion(func, target, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createHloFusionToLinalgPass(llvm::StringRef anchorTag,
                                  bool enablePrimitiveOps) {
  return std::make_unique<HloFusionToLinalgPass>(anchorTag, enablePrimitiveOps);
}
