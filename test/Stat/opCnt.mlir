// RUN: byteir-stat -op-cnt %s | FileCheck %s

module {
  func @tf_add(%arg0 : tensor<2x4xf32>, %arg1 : tensor<2x4xf32>) -> (tensor<2x4xf32>) {
    %0 = "tf.Add"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %1 = "tf.Mul"(%0, %arg0) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = "mhlo.add"(%1, %arg0) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %3 = "mhlo.add"(%1, %2) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %3 : tensor<2x4xf32>
  }
// CHECK: builtin.func 1
// CHECK: mhlo.add 2
// CHECK: std.return 1
// CHECK: tf.Add 1
// CHECK: tf.Mul 1
}

