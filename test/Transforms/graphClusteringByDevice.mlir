// RUN: byteir-opt %s -allow-unregistered-dialect -graph-clustering-by-device="attr-name=device device=cuda" | FileCheck %s

module {
  func @contain_string(%arg0: tensor<1x1x!tf_type.string>, %arg1: tensor<1x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %cst = "tf.Const"() {device = "host", value = dense<"fork_active_pay"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %0 = "tf.Equal"(%arg0, %cst) {device = "host", incompatible_shape_error = true} : (tensor<1x1x!tf_type.string>, tensor<!tf_type.string>) -> tensor<1x1xi1>
    %1 = "mhlo.reshape"(%0) : (tensor<1x1xi1>) -> tensor<i1>
    %2 = "mhlo.select"(%1, %arg1, %arg2) : (tensor<i1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %3 = mhlo.add %2, %arg1 : tensor<1x10xf32>
    return %3 : tensor<1x10xf32>
  }
// CHECK-LABEL: func @contain_string(%arg0: tensor<1x1x!tf_type.string>, %arg1: tensor<1x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> {
// CHECK-NEXT: %[[RES0:.*]] = call @contain_string_host(%arg0) : (tensor<1x1x!tf_type.string>) -> tensor<1x1xi1>
// CHECK-NEXT: %[[RES1:.*]] = call @contain_string_cuda(%0, %arg1, %arg2) : (tensor<1x1xi1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>

// CHECK: func @contain_string_host(%arg0: tensor<1x1x!tf_type.string>) -> tensor<1x1xi1> attributes {device = "host"} {
// CHECK-NEXT: %[[RES0:.*]] = "tf.Const"() {device = "host", value = dense<"fork_active_pay"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
// CHECK-NEXT: %[[RES1:.*]] = "tf.Equal"(%arg0, %0) {device = "host", incompatible_shape_error = true} : (tensor<1x1x!tf_type.string>, tensor<!tf_type.string>) -> tensor<1x1xi1>

// CHECK: func @contain_string_cuda(%arg0: tensor<1x1xi1>, %arg1: tensor<1x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> attributes {device = "cuda"} {
// CHECK-NEXT: %[[RES0:.*]] = "mhlo.reshape"(%arg0) : (tensor<1x1xi1>) -> tensor<i1>
// CHECK-NEXT: %[[RES1:.*]] = "mhlo.select"(%0, %arg1, %arg2) : (tensor<i1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
// CHECK-NEXT: %[[RES2:.*]] = mhlo.add %1, %arg1 : tensor<1x10xf32>

  func @no_host_ops(%arg0: tensor<1x1x!tf_type.string>, %arg1: tensor<1x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %cst = "tf.Const"() {value = dense<"fork_active_pay"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %0 = "tf.Equal"(%arg0, %cst) {incompatible_shape_error = true} : (tensor<1x1x!tf_type.string>, tensor<!tf_type.string>) -> tensor<1x1xi1>
    %1 = "mhlo.reshape"(%0) : (tensor<1x1xi1>) -> tensor<i1>
    %2 = "mhlo.select"(%1, %arg1, %arg2) : (tensor<i1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %3 = mhlo.add %2, %arg1 : tensor<1x10xf32>
    return %3 : tensor<1x10xf32>
  }
// CHECK-LABEL:  func @no_host_ops(%arg0: tensor<1x1x!tf_type.string>, %arg1: tensor<1x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> {
// CHECK-NEXT: %0 = call @no_host_ops_cuda(%arg0, %arg1, %arg2) : (tensor<1x1x!tf_type.string>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
// CHECK-NEXT: return %0 : tensor<1x10xf32>
}

