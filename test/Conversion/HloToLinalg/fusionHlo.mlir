// RUN: byteir-opt -hlo-fusion-to-linalg -linalg-fuse-elementwise-ops %s | FileCheck %s -check-prefix=NOTAG
// RUN: byteir-opt -hlo-fusion-to-linalg="anchor-tag="test"" -linalg-fuse-elementwise-ops %s | FileCheck %s -check-prefix=TESTTAG

// NOTAG-LABEL: fusion_broadcast_tag
// TESTTAG-LABEL: fusion_broadcast_tag
func.func @fusion_broadcast_tag(%arg0: tensor<6x12x96xf32>, %arg1: tensor<6x12x96x96xf32>) -> tensor<6x12x96x96xf32> attributes {test} {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<6x12x96xf32>) -> tensor<6x12x96x96xf32>
  %1 = mhlo.subtract %arg1, %0 : tensor<6x12x96x96xf32>
  %2 = "mhlo.exponential"(%1) : (tensor<6x12x96x96xf32>) -> tensor<6x12x96x96xf32>
  return %2 : tensor<6x12x96x96xf32>
  // NOTAG: linalg.generic
  // NOTAG: arith.subf
  // NOTAG-NEXT: math.exp
  // NOTAG-NEXT: linalg.yield
  // TESTTAG: linalg.generic
  // TESTTAG: arith.subf
  // TESTTAG-NEXT: math.exp
  // TESTTAG-NEXT: linalg.yield
}

// NOTAG-LABEL: fusion_broadcast_notag
// TESTTAG-LABEL: fusion_broadcast_notag
func.func @fusion_broadcast_notag(%arg0: tensor<6x12x96xf32>, %arg1: tensor<6x12x96x96xf32>) -> tensor<6x12x96x96xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<6x12x96xf32>) -> tensor<6x12x96x96xf32>
  %1 = mhlo.subtract %arg1, %0 : tensor<6x12x96x96xf32>
  %2 = "mhlo.exponential"(%1) : (tensor<6x12x96x96xf32>) -> tensor<6x12x96x96xf32>
  return %2 : tensor<6x12x96x96xf32>
  // NOTAG: linalg.generic
  // NOTAG: arith.subf
  // NOTAG-NEXT: math.exp
  // NOTAG-NEXT: linalg.yield
  // TESTTAG: mhlo.broadcast_in_dim
  // TESTTAG-NEXT: mhlo.subtract
  // TESTTAG-NEXT: mhlo.exponential
  // TESTTAG-NEXT: return
}