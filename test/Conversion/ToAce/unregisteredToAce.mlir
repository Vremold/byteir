// RUN: byteir-opt --convert-unregistered-to-ace -allow-unregistered-dialect %s | FileCheck %s

module {
  func @tf_add(%arg0 : tensor<2x4xf32>, %arg1 : tensor<2x4xf32>) -> (tensor<2x4xf32>) {
    %0 = "tf.Add"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %1 = "tf.Mul"(%0, %arg0) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = "mhlo.add"(%1, %arg0) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %2 : tensor<2x4xf32>
  }

// CHECK: func @tf_add(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
// CHECK:   %0 = "ace.opaque"(%arg0, %arg1) ({
// CHECK:   ^bb0(%arg2: tensor<2x4xf32>, %arg3: tensor<2x4xf32>):
// CHECK:     %3 = "tf.Add"(%arg2, %arg3) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:     "ace.return"(%3) : (tensor<2x4xf32>) -> ()
// CHECK:   }) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:   %1 = "ace.opaque"(%0, %arg0) ({
// CHECK:   ^bb0(%arg2: tensor<2x4xf32>, %arg3: tensor<2x4xf32>):
// CHECK:     %3 = "tf.Mul"(%arg2, %arg3) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:     "ace.return"(%3) : (tensor<2x4xf32>) -> ()
// CHECK:   }) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:   %2 = mhlo.add %1, %arg0 : tensor<2x4xf32>
// CHECK:   return %2 : tensor<2x4xf32>
}
