//===- LinalgExtTransformOps.h - Linalg transform ops ----------*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMOPS_LINALGEXTTRANSFORMOPS_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMOPS_LINALGEXTTRANSFORMOPS_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
class TilingInterface;
class RewriterBase;
namespace linalg {
class GenericOp;
class LinalgOp;
} // namespace linalg
} // namespace mlir

//===----------------------------------------------------------------------===//
// LinalgExt Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace linalg_ext {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace linalg_ext
} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMOPS_LINALGEXTTRANSFORMOPS_H
