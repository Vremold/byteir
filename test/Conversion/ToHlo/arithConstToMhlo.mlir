// RUN: byteir-opt %s -convert-arith-to-mhlo | FileCheck %s

// CHECK-LABEL: func @const
func @const() -> tensor<4x4xf32> {
  // CHECK: mhlo.constant
  %0 = arith.constant dense<0.000000e+00> : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
