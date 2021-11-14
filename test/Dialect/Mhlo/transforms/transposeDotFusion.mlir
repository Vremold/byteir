// RUN: byteir-opt %s -fuse-transpose-dot | FileCheck %s

func @lhs_transpose_dot(%arg0 : tensor<64x128xf32>, %arg1 : tensor<64x32xf32>) -> tensor<128x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<64x128xf32>) -> tensor<128x64xf32>
    %1 = "mhlo.dot"(%0, %arg1) : (tensor<128x64xf32>, tensor<64x32xf32>) -> tensor<128x32xf32>
    return %1 : tensor<128x32xf32>
}
// CHECK-LABEL: func @lhs_transpose_dot
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:  mhlo.transpose
// CHECK-NEXT:  mhlo.dot
// CHECK-NEXT:  mhlo.return
// CHECK-NEXT:  }) {__byre__lhs_contracting_dimension = 0 : i64, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"}
// CHECK:  return

func @rhs_transpose_dot(%arg0 : tensor<128x64xf32>, %arg1 : tensor<32x64xf32>) -> tensor<128x32xf32> {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<32x64xf32>) -> tensor<64x32xf32>
    %1 = "mhlo.dot"(%arg0, %0) : (tensor<128x64xf32>, tensor<64x32xf32>) -> tensor<128x32xf32>
    return %1 : tensor<128x32xf32>
}
// CHECK-LABEL: func @rhs_transpose_dot
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:  mhlo.transpose
// CHECK-NEXT:  mhlo.dot
// CHECK-NEXT:  mhlo.return
// CHECK-NEXT:  }) {__byre__lhs_contracting_dimension = 1 : i64, __byre__rhs_contracting_dimension = 1 : i64, byre_compute_name = "MatmulOp"}
// CHECK:  return

func @lhs_rhs_transpose_dot(%arg0 : tensor<64x128xf32>, %arg1 : tensor<32x64xf32>) -> tensor<128x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<64x128xf32>) -> tensor<128x64xf32>
    %1 = "mhlo.transpose"(%arg1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<32x64xf32>) -> tensor<64x32xf32>
    %2 = "mhlo.dot"(%0, %1) : (tensor<128x64xf32>, tensor<64x32xf32>) -> tensor<128x32xf32>
    return %2 : tensor<128x32xf32>
}
// CHECK-LABEL: func @lhs_rhs_transpose_dot
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:  mhlo.transpose
// CHECK-NEXT:  mhlo.transpose
// CHECK-NEXT:  mhlo.dot
// CHECK-NEXT:  mhlo.return
// CHECK-NEXT:  }) {__byre__lhs_contracting_dimension = 0 : i64, __byre__rhs_contracting_dimension = 1 : i64, byre_compute_name = "MatmulOp"}
// CHECK:  return
