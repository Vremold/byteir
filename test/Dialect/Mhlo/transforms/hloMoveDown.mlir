// RUN: byteir-opt %s -hlo-move-down | FileCheck %s
// RUN: byteir-opt %s -hlo-move-down="multi-user" | FileCheck %s --check-prefix MULTIUSER
// RUN: byteir-opt %s -hlo-move-down="all-multi-user" | FileCheck %s --check-prefix AllMULTIUSER

func @transpose_move_down_unary(%arg0 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    return %1 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func @transpose_move_down_unary
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: return

func @transpose_binary_same(%arg0 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = mhlo.add %0, %0 : tensor<20x31x32xf32>
    return %1 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func @transpose_binary_same
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: return

func @transpose_move_down_binary_splat_const(%arg0 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<20x31x32xf32>
    %1 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %2 = mhlo.add %1, %0 : tensor<20x31x32xf32>
    return %2 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func @transpose_move_down_binary_splat_const
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: return

func @transpose_move_down_unary_and_cancel(%arg0 : tensor<31x20x32xf32>) -> tensor<31x20x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = "mhlo.transpose"(%1) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<20x31x32xf32>) -> tensor<31x20x32xf32>
    return %2 : tensor<31x20x32xf32>
}
// CHECK-LABEL: func @transpose_move_down_unary_and_cancel
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: return

func @transpose_move_down_unary_many_and_cancel(%arg0 : tensor<31x20x32xf32>) -> tensor<31x20x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = "mhlo.sine"(%1) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %3 = "mhlo.sine"(%2) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %4 = "mhlo.sine"(%3) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %5 = "mhlo.transpose"(%4) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<20x31x32xf32>) -> tensor<31x20x32xf32>
    return %5 : tensor<31x20x32xf32>
}
// CHECK-LABEL: func @transpose_move_down_unary_many_and_cancel
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: return


func @transpose_move_down_two_unary(%arg0 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = "mhlo.sine"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %3 = mhlo.add %1, %2 : tensor<20x31x32xf32>
    return %3 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func @transpose_move_down_two_unary
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: return

// MULTIUSER-LABEL: func @transpose_move_down_two_unary
// MULTIUSER-DAG{ABS}: mhlo.abs
// MULTIUSER-NEXT{ABS}: mhlo.transpose
// MULTIUSER-DAG{SINE}: mhlo.sine
// MULTIUSER-NEXT{SINE}: mhlo.transpose
// MULTIUSER: mhlo.add
// MULTIUSER-NEXT: return

// AllMULTIUSER-LABEL: func @transpose_move_down_two_unary
// AllMULTIUSER-DAG{ABS}: mhlo.abs
// AllMULTIUSER-NEXT{ABS}: mhlo.transpose
// AllMULTIUSER-DAG{SINE}: mhlo.sine
// AllMULTIUSER-NEXT{SINE}: mhlo.transpose
// AllMULTIUSER: mhlo.add
// AllMULTIUSER-NEXT: return

func @transpose_move_down_1_unary_1_invalid(%arg0 : tensor<31x20x32xf32>, %arg1 : tensor<20x31x32xf32>)-> tensor<20x31x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = mhlo.add %0, %arg1 : tensor<20x31x32xf32>
    %3 = mhlo.multiply %1, %2 : tensor<20x31x32xf32>
    return %3 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func @transpose_move_down_1_unary_1_invalid
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return

// MULTIUSER-LABEL: func @transpose_move_down_1_unary_1_invalid
// MULTIUSER-DAG{ABS}: mhlo.abs
// MULTIUSER-NEXT{ABS}: mhlo.transpose
// MULTIUSER-DAG{ADD}: mhlo.transpose
// MULTIUSER-NEXT{ADD}: mhlo.add
// MULTIUSER: mhlo.multiply
// MULTIUSER-NEXT: return

// AllMULTIUSER-LABEL: func @transpose_move_down_1_unary_1_invalid
// AllMULTIUSER-NEXT: mhlo.transpose
// AllMULTIUSER-NEXT: mhlo.abs
// AllMULTIUSER-NEXT: mhlo.add
// AllMULTIUSER-NEXT: mhlo.multiply
// AllMULTIUSER-NEXT: return

func @reshape_move_down_unary(%arg0 : tensor<31x20x32xf32>) -> tensor<31x640xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x640xf32>
    %1 = "mhlo.abs"(%0) : (tensor<31x640xf32>) -> tensor<31x640xf32>
    return %1 : tensor<31x640xf32>
}
// CHECK-LABEL: func @reshape_move_down_unary
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

func @reshape_not_move_down_convert(%142: tensor<1x32xi32>) -> tensor<1x32x32xf32> {
    %144 = "mhlo.reshape"(%142) : (tensor<1x32xi32>) -> tensor<1x1x32xi32>
    %145 = "mhlo.convert"(%144) : (tensor<1x1x32xi32>) -> tensor<1x1x32xf32>
    %179 = "mhlo.broadcast_in_dim"(%145) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x32xf32>) -> tensor<1x32x32xf32>
    return %179 : tensor<1x32x32xf32>
}
// CHECK-LABEL: func @reshape_not_move_down_convert
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.convert
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: return

func @reshape_binary_same(%arg0 : tensor<31x20x32xf32>) -> tensor<31x640xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x640xf32>
    %1 = mhlo.add %0, %0 : tensor<31x640xf32>
    return %1 : tensor<31x640xf32>
}
// CHECK-LABEL: func @reshape_binary_same
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

func @reshape_move_down_binary_splat_const(%arg0 : tensor<31x20x32xf32>) -> tensor<31x640xf32> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<31x640xf32>
    %1 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x640xf32>
    %2 = mhlo.add %1, %0 : tensor<31x640xf32>
    return %2 : tensor<31x640xf32>
}
// CHECK-LABEL: func @reshape_move_down_binary_splat_const
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

func @reshape_move_down_unary_and_cancel(%arg0 : tensor<31x20x32xf32>) -> tensor<31x20x32xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x640xf32>
    %1 = "mhlo.abs"(%0) : (tensor<31x640xf32>) -> tensor<31x640xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<31x640xf32>) -> tensor<31x20x32xf32>
    return %2 : tensor<31x20x32xf32>
}
// CHECK-LABEL: func @reshape_move_down_unary_and_cancel
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: return

func @reshape_move_down_unary_many_and_cancel(%arg0 : tensor<31x20x32xf32>) -> tensor<31x20x32xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<31x640xf32>
    %1 = "mhlo.abs"(%0) : (tensor<31x640xf32>) -> tensor<31x640xf32>
    %2 = "mhlo.sine"(%1) : (tensor<31x640xf32>) -> tensor<31x640xf32>
    %3 = "mhlo.sine"(%2) : (tensor<31x640xf32>) -> tensor<31x640xf32>
    %4 = "mhlo.sine"(%3) : (tensor<31x640xf32>) -> tensor<31x640xf32>
    %5 = "mhlo.reshape"(%4) : (tensor<31x640xf32>) -> tensor<31x20x32xf32>
    return %5 : tensor<31x20x32xf32>
}
// CHECK-LABEL: func @reshape_move_down_unary_many_and_cancel
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: return

func @reshape_move_down_two_unary(%arg0 : tensor<31x20x32xf32>) -> tensor<20x31x32xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = "mhlo.sine"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %3 = mhlo.add %1, %2 : tensor<20x31x32xf32>
    return %3 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func @reshape_move_down_two_unary
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.sine
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: return

// MULTIUSER-LABEL: func @reshape_move_down_two_unary
// MULTIUSER-DAG{ABS}: mhlo.abs
// MULTIUSER-NEXT{ABS}: mhlo.reshape
// MULTIUSER-DAG{SINE}: mhlo.sine
// MULTIUSER-NEXT{SINE}: mhlo.reshape
// MULTIUSER: mhlo.add
// MULTIUSER-NEXT: return

// AllMULTIUSER-LABEL: func @reshape_move_down_two_unary
// AllMULTIUSER-DAG{ABS}: mhlo.abs
// AllMULTIUSER-NEXT{ABS}: mhlo.reshape
// AllMULTIUSER-DAG{SINE}: mhlo.sine
// AllMULTIUSER-NEXT{SINE}: mhlo.reshape
// AllMULTIUSER: mhlo.add
// AllMULTIUSER-NEXT: return

func @reshape_move_down_1_unary_1_invalid(%arg0 : tensor<31x20x32xf32>, %arg1 : tensor<20x31x32xf32>)-> tensor<20x31x32xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.abs"(%0) : (tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %2 = mhlo.add %0, %arg1 : tensor<20x31x32xf32>
    %3 = mhlo.multiply %1, %2 : tensor<20x31x32xf32>
    return %3 : tensor<20x31x32xf32>
}
// CHECK-LABEL: func @reshape_move_down_1_unary_1_invalid
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.add
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return

// MULTIUSER-LABEL: func @reshape_move_down_1_unary_1_invalid
// MULTIUSER-DAG{ABS}: mhlo.abs
// MULTIUSER-NEXT{ABS}: mhlo.reshape
// MULTIUSER-DAG{ADD}: mhlo.reshape
// MULTIUSER-NEXT{ADD}: mhlo.add
// MULTIUSER: mhlo.multiply
// MULTIUSER-NEXT: return

// AllMULTIUSER-LABEL: func @reshape_move_down_1_unary_1_invalid
// AllMULTIUSER-NEXT: mhlo.reshape
// AllMULTIUSER-NEXT: mhlo.abs
// AllMULTIUSER-NEXT: mhlo.add
// AllMULTIUSER-NEXT: mhlo.multiply
// AllMULTIUSER-NEXT: return

