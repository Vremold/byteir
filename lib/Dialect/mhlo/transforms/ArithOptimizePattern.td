#ifndef MHLO_ARITH_OPT_PATTERN
#define MHLO_ARITH_OPT_PATTERN

include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"

def isZeroValue : Constraint<CPred<"isSplatValue($0.dyn_cast<DenseIntElementsAttr>(), 0) || isSplatValue($0.dyn_cast<DenseFPElementsAttr>(), 0.0)">, "value is zero">;
def isOneValue : Constraint<CPred<"isSplatValue($0.dyn_cast<DenseIntElementsAttr>(), 1) || isSplatValue($0.dyn_cast<DenseFPElementsAttr>(), 1.0)">, "value is one">;

def AddZeroLeftPattern : Pat<
    (HLO_AddOp
      (HLO_ConstOp $value),
      $rhs
    ),
    (replaceWithValue $rhs),
    [(isZeroValue $value)]>;

def AddZeroRightPattern : Pat<
    (HLO_AddOp
      $lhs,
      (HLO_ConstOp $value)
    ),
    (replaceWithValue $lhs),
    [(isZeroValue $value)]>;

def MulOneLeftPattern : Pat<
    (HLO_MulOp
        (HLO_ConstOp $value),
        $rhs
    ),
    (replaceWithValue $rhs),
    [(isOneValue $value)]>;

def MulOneRightPattern : Pat<
    (HLO_MulOp
        $lhs,
        (HLO_ConstOp $value)
    ),
    (replaceWithValue $lhs),
    [(isOneValue $value)]>;

#endif // MHLO_ARITH_OPT_PATTERN
