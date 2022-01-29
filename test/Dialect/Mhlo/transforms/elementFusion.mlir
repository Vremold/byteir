// RUN: byteir-opt %s -fuse-element | FileCheck %s -check-prefix=NOTAG
// RUN: byteir-opt %s -fuse-element="attach-tag=test" | FileCheck %s -check-prefix=TESTTAG

func @mhlo_element(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "mhlo.abs"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = "mhlo.add"(%arg0, %arg2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %3 = "mhlo.add"(%1, %2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %3 : tensor<4xf32>
}
// NOTAG-LABEL: func @mhlo_element
// NOTAG-NEXT:  mhlo.fusion
// NOTAG-NEXT:  mhlo.add
// NOTAG-NEXT:  mhlo.abs
// NOTAG-NEXT:  mhlo.add
// NOTAG-NEXT:  mhlo.add
// NOTAG-NEXT:  mhlo.return
// NOTAG-NOT: {test}
// NOTAG:  return

// TESTTAG-LABEL: func @mhlo_element
// TESTTAG-NEXT:  mhlo.fusion
// TESTTAG-NEXT:  mhlo.add
// TESTTAG-NEXT:  mhlo.abs
// TESTTAG-NEXT:  mhlo.add
// TESTTAG-NEXT:  mhlo.add
// TESTTAG-NEXT:  mhlo.return
// TESTTAG: {test}
// TESTTAG:  return


func @mhlo_element_broadcast(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>, %arg2 : tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<3x4xf32>
  %2 = "mhlo.abs"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = "mhlo.add"(%1, %arg2) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  %4 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<3x4xf32>
  %5 = "mhlo.add"(%3, %4) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %5 : tensor<3x4xf32>
}
// NOTAG-LABEL: func @mhlo_element_broadcast
// NOTAG-NEXT:  mhlo.fusion
// NOTAG-NEXT:  mhlo.add
// NOTAG-NEXT:  mhlo.abs
// NOTAG:  mhlo.fusion
// NOTAG-NEXT:  mhlo.broadcast_in_dim
// NOTAG-NEXT:  mhlo.add
// NOTAG-NEXT:  mhlo.broadcast_in_dim
// NOTAG-NEXT:  mhlo.add
// NOTAG-NEXT:  mhlo.return
// NOTAG-NOT: {test}
// NOTAG:  return

// TESTTAG-LABEL: func @mhlo_element_broadcast
// TESTTAG-NEXT:  mhlo.fusion
// TESTTAG-NEXT:  mhlo.add
// TESTTAG-NEXT:  mhlo.abs
// TESTTAG:  mhlo.fusion
// TESTTAG-NEXT:  mhlo.broadcast_in_dim
// TESTTAG-NEXT:  mhlo.add
// TESTTAG-NEXT:  mhlo.broadcast_in_dim
// TESTTAG-NEXT:  mhlo.add
// TESTTAG-NEXT:  mhlo.return
// TESTTAG: {test}
// TESTTAG:  return


func @mhlo_element_reshape(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>, %arg2 : tensor<2x2xf32>) -> tensor<4xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "mhlo.abs"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = "mhlo.reshape"(%arg2) : (tensor<2x2xf32>) -> tensor<4xf32>
  %3 = "mhlo.add"(%0, %2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %3 : tensor<4xf32>
}
// NOTAG-LABEL: func @mhlo_element_reshape
// NOTAG:  mhlo.fusion
// NOTAG-NEXT:  mhlo.add
// NOTAG-NEXT:  mhlo.abs
// NOTAG-NEXT:  mhlo.reshape
// NOTAG-NEXT:  mhlo.add
// NOTAG-NOT: {test}
// NOTAG:  return

// TESTTAG-LABEL: func @mhlo_element_reshape
// TESTTAG:  mhlo.fusion
// TESTTAG-NEXT:  mhlo.add
// TESTTAG-NEXT:  mhlo.abs
// TESTTAG-NEXT:  mhlo.reshape
// TESTTAG-NEXT:  mhlo.add
// TESTTAG: {test}
// TESTTAG:  return

func @shared_constant(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>, %arg3 : tensor<4x4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<4xf32>
  %1 = "mhlo.add"(%arg0, %0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %2 = "mhlo.abs"(%1) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %4 = "mhlo.add"(%1, %3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %5 = "mhlo.dot"(%arg3, %0) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %4, %5 : tensor<4xf32>, tensor<4xf32>
}
// NOTAG-LABEL: func @shared_constant
// NOTAG:  mhlo.fusion
// NOTAG-NEXT:  mhlo.constant
// NOTAG-NEXT:  mhlo.add
// NOTAG-NEXT:  mhlo.abs
// NOTAG-NEXT:  mhlo.add
// NOTAG-NEXT:  mhlo.add
// NOTAG-NOT: {test}
// NOTAG:  return

// TESTTAG-LABEL: func @shared_constant
// TESTTAG:  mhlo.fusion
// TESTTAG-NEXT:  mhlo.constant
// TESTTAG-NEXT:  mhlo.add
// TESTTAG-NEXT:  mhlo.abs
// TESTTAG-NEXT:  mhlo.add
// TESTTAG-NEXT:  mhlo.add
// TESTTAG: {test}
// TESTTAG:  return

func private @empty(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>, %arg2 : tensor<3x4xf32>) -> tensor<3x4xf32>
// NOTAG-LABEL: func private @empty
// TESTTAG-LABEL: func private @empty
