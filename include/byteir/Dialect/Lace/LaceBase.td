//===-- LaceBase.td - Lace dialect definitions ---------*----- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the Lace dialect
//
//===----------------------------------------------------------------------===//

#ifndef LACE_BASE
#define LACE_BASE

include "byteir/Dialect/Ace/AceBase.td" // include Ace types
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Lace Dialect.
//===----------------------------------------------------------------------===//

// Lace Dialect works as an extension to lmhlo dialect, these two dialects work
// together as high level IR in byteir.
def Lace_Dialect : Dialect {
  let name = "lace";
  let cppNamespace = "::mlir::lace";

  let emitAccessorPrefix = kEmitAccessorPrefix_Both; // TODO: change to prefix
}

//===----------------------------------------------------------------------===//
// Lace Types.
//===----------------------------------------------------------------------===//

def Lace_StringType : TypeAlias<Ace_StringType, "string type in lace dialect">;

def Lace_BufferType : MemRefOf<[AnyFloat, AnyInteger, Lace_StringType]>;

//===----------------------------------------------------------------------===//
// Lace OpInterfaces.
//===----------------------------------------------------------------------===//

def Lace_AliasLikeOpInterface : OpInterface<"AliasLikeOpInterface"> {
  let cppNamespace = "::mlir::lace";

  let methods = [
    InterfaceMethod<[{
        Query the offset between the target memref and source memref
      }], "int64_t", "getOffsetElem"
    >,
  ];
}

#endif // LACE_BASE
