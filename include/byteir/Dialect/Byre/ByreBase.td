//===-- ByreBase.td - Byre dialect definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the Byre dialect
//
//===----------------------------------------------------------------------===//

#ifndef BYRE_BASE
#define BYRE_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Byre Dialect.
//===----------------------------------------------------------------------===//

def Byre_Dialect : Dialect {
  let name = "byre";
  let cppNamespace = "::mlir::byre";
  let hasOperationAttrVerify = 1;
  let useDefaultTypePrinterParser = 0;

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate the modules that contain
    /// kernel modules.
    static StringRef getContainerModuleAttrName() {
      return "byre.container_module";
    }

    static StringRef getModuleMemorySpaceAttrName() {
      return "byre.memory_space";
    }

    static StringRef getEntryPointFunctionAttrName() {
      return "byre.entry_point";
    }

    static StringRef getEntryPointFuncArgTypeAttrName() {
      return "byre.argtype";
    }

    static StringRef getEntryPointFuncArgNameAttrName() {
      return "byre.argname";
    }

    static StringRef getEntryPointFuncResultAttrsName() {
      return "byre.result_attrs";
    }

    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

    void printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

def Byre_AsyncToken : DialectType<
  Byre_Dialect, CPred<"$_self.isa<::mlir::byre::AsyncTokenType>()">, "async token type">,
             BuildableType<"mlir::byre::AsyncTokenType::get($_builder.getContext())">;

def Byre_AsyncOpInterface : OpInterface<"AsyncOpInterface"> {
  let description = [{
    Interface for Runtime operations that execute asynchronously on the device.

    The op doesn't start executing until all depent ops producing the async
    dependency tokens have finished executing.

    If the op returns a token, the op merely schedules the execution on the
    device and returns immediately, without waiting for the execution to
    complete. On the hand, if the op does not return a token, the op will wait
    for the execution to complete.
  }];
  let cppNamespace = "::mlir::byre";

  let methods = [
    InterfaceMethod<[{
        Query the operands that represent async dependency tokens.
      }],
      "OperandRange", "getAsyncDependencies", (ins), [{}], [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncDependencies();
      }]
    >,
    InterfaceMethod<[{
        Adds a new token to the list of async dependencies.
      }],
      "void", "addAsyncDependency", (ins "Value":$token),
      [{}], [{
        ::mlir::byre::addAsyncDependency(this->getOperation(), token);
      }]
    >,
    InterfaceMethod<[{
        Query the result that represents the async token to depend on.
      }],
      "OpResult", "getAsyncToken", (ins), [{}], [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncToken().template dyn_cast_or_null<OpResult>();
      }]
    >
  ];
}

// The 'ByreInterface' provides access to the 'ByreOp' interface.
def ByreInterface : OpInterface<"ByreOp"> {
  let cppNamespace = "::mlir::byre";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the name of callee for this op.
      }],
      /*retTy=*/"std::string",
      /*methodName=*/"getCalleeName",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Operation* op = $_op.getOperation();
        if (auto flatSymAttr = op->getAttrOfType<::mlir::FlatSymbolRefAttr>("callee")) {
          return flatSymAttr.getValue().str();
        } else if (auto strAttr = op->getAttrOfType<::mlir::StringAttr>("callee")) {
          return strAttr.str();
        }
        return "";
      }]
    >
  ];

}

def Byre_ArgType_None : I32BitEnumAttrCaseNone<"None">;            // 0
def Byre_ArgType_Input : I32BitEnumAttrCaseBit<"Input", 0x0000>;   // 1
def Byre_ArgType_Output : I32BitEnumAttrCaseBit<"Output", 0x0001>; // 2
def Byre_ArgType_Weight : I32BitEnumAttrCaseBit<"Weight", 0x0002>; // 4

def Byre_ArgTypeAttr : I32BitEnumAttr<
    "EntryFuncArgType",
    "entry function argument type", [
        Byre_ArgType_None,
        Byre_ArgType_Input,
        Byre_ArgType_Output,
        Byre_ArgType_Weight
    ]> {
  let cppNamespace = "::mlir::byre";
}

#endif // BYRE_BASE
