// RUN: byteir-opt %s -allow-unregistered-dialect -graph-clustering-by-device="attr-name=device device=cuda" -canonicalize | FileCheck %s

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

  func @duplicate_splat_mhlo_const(%arg0: tensor<!tf_type.string>, %arg1: tensor<i1>, %arg2: tensor<i1>) -> tensor<i1> {
    %cst = "tf.Const"() {device = "host", value = dense<"fork_active_pay"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %0 = "tf.Equal"(%arg0, %cst) {device = "host", incompatible_shape_error = true} : (tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<i1>
    %1 = mhlo.constant dense<true> : tensor<i1>
    %2 = "mhlo.add"(%0, %1) {device = "host"} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = "mhlo.select"(%2, %arg1, %arg2) : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = mhlo.add %3, %1 : tensor<i1>
    return %4 : tensor<i1>
  }
// CHECK-LABEL: func @duplicate_splat_mhlo_const(%arg0: tensor<!tf_type.string>, %arg1: tensor<i1>, %arg2: tensor<i1>) -> tensor<i1> {
// CHECK_NEXT:    %0 = call @duplicate_splat_mhlo_const_host(%arg0) : (tensor<!tf_type.string>) -> tensor<i1>
// CHECK_NEXT:    %1 = call @duplicate_splat_mhlo_const_cuda(%0, %arg1, %arg2) : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
// CHECK_NEXT:    return %1 : tensor<i1>
// CHECK_NEXT:  }

// CHECK-LABEL: func @duplicate_splat_mhlo_const_host(%arg0: tensor<!tf_type.string>) -> tensor<i1> attributes {device = "host"} {
// CHECK-NEXT:   %0 = mhlo.constant dense<true> : tensor<i1>
// CHECK-NEXT:   %1 = "tf.Const"() {device = "host", value = dense<"fork_active_pay"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
// CHECK-NEXT:   %2 = "tf.Equal"(%arg0, %1) {device = "host", incompatible_shape_error = true} : (tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<i1>
// CHECK-NEXT:   %3 = mhlo.add %2, %0 {device = "host"} : tensor<i1>
// CHECK-NEXT:   return %3 : tensor<i1>
// CHECK-NEXT: }

// CHECK-LABEL: func @duplicate_splat_mhlo_const_cuda(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<i1>) -> tensor<i1> attributes {device = "cuda"} {
// CHECK-NEXT:   %0 = mhlo.constant dense<true> : tensor<i1>
// CHECK-NEXT:   %1 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
// CHECK-NEXT:   %2 = mhlo.add %1, %0 : tensor<i1>
// CHECK-NEXT:   return %2 : tensor<i1>
// CHECK-NEXT: }
}

