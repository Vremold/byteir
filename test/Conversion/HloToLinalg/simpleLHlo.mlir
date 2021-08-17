// RUN: byteir-opt -lhlo-legalize-to-linalg %s | FileCheck %s

// CHECK-LABEL: lmhlo_add
func @lmhlo_add(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
                   %result: memref<2x2xf32>) {
  "lmhlo.add"(%lhs, %rhs, %result)
      : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
  // CHECK: linalg.generic
  // CHECK: addf
}