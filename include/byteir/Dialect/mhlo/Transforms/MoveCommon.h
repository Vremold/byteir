//===- MoveCommon.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_MOVECOMMON_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_MOVECOMMON_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;

// common Move patterns template for some general cases



template <typename OTy> 
struct HloMoveDownPattern : public OpRewritePattern<OTy> {
  HloMoveDownPattern(MLIRContext *ctx,
                 const llvm::DenseSet<llvm::StringRef> &blocker,
                 bool supportAllMultiUser = false,
                 bool supportMultiUser = false)
      : OpRewritePattern<OTy>(ctx), 
        blockers(blocker),
        allMultiUser(supportAllMultiUser),
        multiUser(supportMultiUser) {}

  const llvm::DenseSet<llvm::StringRef>& blockers;
  bool allMultiUser; // allow transposed result used in multiple users, all of
                     // them must be legal
  bool multiUser;    // allow transposed result used in multiple users, including some illegal ops

/*
//  multiUser  = false,  AllMultiUser = false
case 1
          S
          |
        OTy
        /   \
      A     B
==> no transformation
*/

/*
//  multiUser  = true,  AllMultiUser = false
case 2
          S
          |
          OTy
        /   \
        A     B
==>       S
          /  \
        A    B
        |    |
        OTy  OTy 
case 3
          S
          |
          OTy
        /  |  \
        A   B   IllegalC
==>       S
          / |  \
        A  B  OTy
        |  |   |
        OTy OTy IllegalC
*/

/*
//  AllMultiUser = true,  multiUser = true/false/don't care
  case 2
          S
          |
          OTy
        /  |  \
        A   B   IllegalC
==> no transformation
*/
};

template <typename OTy>
struct HloMoveUpPattern : public OpRewritePattern<OTy> {
  HloMoveUpPattern(MLIRContext *ctx,
                   const llvm::DenseSet<llvm::StringRef> &blocker,
                   bool supportMultiInput = false)
      : OpRewritePattern<OTy>(ctx), 
        blockers(blocker),
        multiInput(supportMultiInput) {}

  const llvm::DenseSet<llvm::StringRef> &blockers;
  bool multiInput;    // allow producer of transpose has multiple inputs
/*
//  multiInput  = false
case 1
         S1    S2
          \   /
            A
            |
            OTy
  ==> no transformation
*/

/*
//  multiInput  = true
case 1
           S1    S2
            \   /
              A
              |
              OTy
=>
          S1    S2
          |     |
          OTy   OTy
            \  /
              A
*/

};


} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_MOVECOMMON_H