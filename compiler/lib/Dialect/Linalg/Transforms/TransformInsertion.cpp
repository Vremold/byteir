//===- TransformInsertion.cpp ---------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Linalg/Transforms/TransformInsertion.h"

#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.h"
#include "byteir/Dialect/Linalg/Utils/TilingUtils.h"
#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/Optional.h"

#include "PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

static int tilingCount = 0;

struct TilingMetadata {
  // Tiling info
  SmallVector<int64_t> tileSizes;
  SmallVector<int64_t> tileInterchange;
  // Tiling annotation for matchOp
  std::string annotation;
};

struct TransformInsertionPass
    : public TransformInsertionBase<TransformInsertionPass> {

  explicit TransformInsertionPass(std::string deviceAnchorName)
      : TransformInsertionBase<TransformInsertionPass>() {
    this->deviceAnchorName = deviceAnchorName;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
    linalg::registerTransformDialectExtension(registry);
  }
};

llvm::Optional<TilingMetadata> getTilingMetadata(Value value) {
  if (!value.getDefiningOp())
    return llvm::None;

  // TODO: use real tile sizes and tile interchange. Get this from Op attr.
  TilingMetadata metadata;
  metadata.tileSizes = {1};
  metadata.tileInterchange = {};
  metadata.annotation =
      getTilingAnnotationPrefix().str() + std::to_string(tilingCount);
  tilingCount++;
  return metadata;
}

void InsertTransformIR(func::FuncOp funcOp, OpBuilder &b) {
  Operation *retOp = funcOp.getBody().front().getTerminator();
  MLIRContext *ctx = b.getContext();

  // only support 1 output for now
  assert(retOp->getNumOperands() == 1);
  Value operand = retOp->getOperand(0);
  auto tilingMetadata = getTilingMetadata(operand);
  if (!tilingMetadata.has_value())
    return;

  auto metadata = tilingMetadata.value();
  auto op = operand.getDefiningOp();
  op->setAttr(metadata.annotation, UnitAttr::get(ctx));
  auto unknownLoc = UnknownLoc::get(ctx);

  auto pdlType = pdl::OperationType::get(ctx);
  SmallVector<Type> resultTypes(1 + metadata.tileSizes.size(), pdlType);

  auto seq = b.create<transform::SequenceOp>(
      unknownLoc, TypeRange(), transform::FailurePropagationMode::Propagate,
      nullptr);
  OpBuilder::InsertionGuard guard(b);

  Block *bodyBlock =
      b.createBlock(&seq.getBody(), seq.getBody().begin(),
                    {pdl::OperationType::get(ctx)}, {unknownLoc});
  b.setInsertionPointToStart(bodyBlock);
  auto annotationAttr = DictionaryAttr::get(
      ctx, b.getNamedAttr(metadata.annotation, UnitAttr::get(ctx)));
  auto match = b.create<transform::MatchOp>(
      unknownLoc, bodyBlock->getArgument(0).getType(),
      bodyBlock->getArgument(0), ArrayAttr(),
      transform::MatchInterfaceEnumAttr(), annotationAttr, TypeAttr());
  b.create<transform::FuseExtOp>(unknownLoc, resultTypes, match,
                                 b.getI64ArrayAttr(metadata.tileSizes),
                                 b.getI64ArrayAttr(metadata.tileInterchange));
  b.create<transform::YieldOp>(unknownLoc);
}

void TransformInsertionPass::runOnOperation() {
  ModuleOp m = getOperation();
  OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());
  for (auto funcOp : m.getOps<func::FuncOp>()) {
    // only tile on device functions
    if (deviceAnchorName != "" && !funcOp->hasAttr(this->deviceAnchorName))
      continue;
    InsertTransformIR(funcOp, builder);
  }
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createTransformInsertionPass(std::string deviceAnchorName) {
  return std::make_unique<TransformInsertionPass>(deviceAnchorName);
}