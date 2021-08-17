//===- MhloToAceActivationPattern.td --------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef MHLO_TO_ACE_ACTIVATION_PATTERN
#define MHLO_TO_ACE_ACTIVATION_PATTERN


include "byteir/Dialect/Ace/AceOps.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"

def CreateAceActivateOp : NativeCodeCall<"$_builder.create<mlir::ace::ActivateOp>($_loc, $0.getType(), $1, $2, $3)">;
def AttrIsZero : Constraint<CPred<"mlir::isZeroAttribute($_self)">>;
def FPDenseSixValue : Constraint<CPred<"mlir::isSplatCloseToValue($_self.dyn_cast<DenseFPElementsAttr>(), 6.0)">>;
def FPDenseSplatValue : Constraint<CPred<"$_self.dyn_cast<DenseFPElementsAttr>() && $_self.dyn_cast<DenseFPElementsAttr>().isSplat()">>;


def MhloToAceReluPattern : Pat<
  (HLO_MaxOp
    $input,
    (HLO_ConstOp
      $value
    )
  ),
  (Ace_ActivateOp $input, (NativeCodeCall<"$_builder.getStringAttr(\"relu\")">)),
  [(AttrIsZero:$value)]>;

def MhloToAceRelu6Pattern : Pat<
  (HLO_ClampOp
    (HLO_ConstOp
      $value1
    ),
    $input,
    (HLO_ConstOp
      $value2
    )
  ),
  (Ace_ActivateOp $input, (NativeCodeCall<"$_builder.getStringAttr(\"relu6\")">)),
  [(AttrIsZero:$value1), (FPDenseSixValue:$value2)]>;

def MhloToAceLeakyReluPattern : Pat<
  (HLO_SelectOp:$output
    (HLO_CompareOp
      $input,
      (HLO_ConstOp $value1),
      HLO_COMPARISON_DIRECTION_GT,
      $_
    ),
    $input,
    (HLO_MulOp
      $input,
      (HLO_ConstOp $value2)
    )
  ),
  (CreateAceActivateOp $output, $input, (NativeCodeCall<"$_builder.getStringAttr(\"leaky_relu\")">),
    (NativeCodeCall<"{$_builder.getNamedAttr(\"alpha\", value2.cast<DenseFPElementsAttr>().getSplatValue<FloatAttr>())}">)),
  [(AttrIsZero:$value1), (FPDenseSplatValue:$value2)]>;

#endif // MHLO_TO_ACE_ACTIVATION_PATTERN