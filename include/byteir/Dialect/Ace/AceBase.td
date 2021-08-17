//===-- AceBase.td - Ace dialect definitions ---------*------- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the Ace dialect
//
//===----------------------------------------------------------------------===//

#ifndef ACE_BASE
#define ACE_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Ace Dialect.
//===----------------------------------------------------------------------===//

// Ace Dialect works as an extension to mhlo dialect, these two dialects work
// together as high level IR in byteir.
def Ace_Dialect : Dialect {
  let name = "ace";
  let cppNamespace = "::mlir::ace";
}

#endif // ACE_BASE
