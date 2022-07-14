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

//===----------------------------------------------------------------------===//
// Ace Type.
//===----------------------------------------------------------------------===//

class Ace_Type<string name, string typeMnemonic> : TypeDef<Ace_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Ace_StringType : Ace_Type<"String", "string"> {
  let description = [{`ace.string` represents string type}];
}

def Ace_ResourceType : Ace_Type<"Resource", "resource"> {
  let description = [{`ace.resource` represents resource type from tensorflow}];
}

def Ace_CustomType : AnyTypeOf<[Ace_StringType, Ace_ResourceType]>;

def Ace_StaticShapeTensor : StaticShapeTensorOf<[Ace_CustomType]>;

//===----------------------------------------------------------------------===//
// Ace Attribute.
//===----------------------------------------------------------------------===//

def TensorEncoding : AttrDef<Ace_Dialect, "TensorEncoding"> {
  let mnemonic = "tensor_encoding";

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$isDynamic
  );

  let summary = "Attribute that extends tensor type.";

  let description = [{
    This attribute is used to extend tensor type with tensor specific
    properties. These properties aren't modeled in the MLIR type. This
    attribute is set in the `encoding` field of the tensor type.

    `is_dynamic` array parameter represents that which dim of tensor is dynamic.
    Sample: tensor<?x16xf32, #ace.tensor_encoding<is_dynamic = [1, 0]>>,
    1 represent the first dimension is dynamic, 0 represent the second
    dimension is not dynamic.

    // TODO(lyq): verify the `is_dynamic` array length is same as shape rank.
  }];
  let assemblyFormat = "`<` `is_dynamic` `=` `[` $isDynamic `]` `>`";

}

#endif // ACE_BASE
