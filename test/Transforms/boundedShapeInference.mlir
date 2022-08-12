// RUN: byteir-opt %s -bounded-shape-infer | FileCheck %s

func.func @SameOperandsAndResultShape(%arg0: tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}, %arg1: tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}) -> tensor<?x4xf32> {
  %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

//CHECK-LABEL: func.func @SameOperandsAndResultShape(%arg0: tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}> {byteir.bounded_shape = [8, 4]}, %arg1: tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}> {byteir.bounded_shape = [8, 4]}) -> tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}> {
//CHECK-NEXT:  %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>
//CHECK-NEXT:  return %0 : tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>

func.func @InferShapedTypeOpInterface(%arg0 : tensor<8x4xi32>, %arg1 : tensor<8x4xi32>) -> tensor<?x4xi1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<?x4xi1>
  return %0 : tensor<?x4xi1>
}

//CHECK-LABEL:func.func @InferShapedTypeOpInterface(%arg0: tensor<8x4xi32>, %arg1: tensor<8x4xi32>) -> tensor<?x4xi1, {byteir.bounded_shape = [8, 4]}> {
//CHECK-NEXT:  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<?x4xi1, {byteir.bounded_shape = [8, 4]}>
//CHECK-NEXT:  return %0 : tensor<?x4xi1, {byteir.bounded_shape = [8, 4]}>

func.func @several_ops(%arg0: tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}, %arg1: tensor<4x4xf32>, %arg2: tensor<4xf32>) -> tensor<?x4xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %1 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %3 = mhlo.add %0, %2 : tensor<?x4xf32>
  return %3 : tensor<?x4xf32>
}

//CHECK-LABEL: func.func @several_ops(%arg0: tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}> {byteir.bounded_shape = [8, 4]}, %arg1: tensor<4x4xf32>, %arg2: tensor<4xf32>) -> tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}> {
//CHECK-NEXT:  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>, tensor<4x4xf32>) -> tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>
//CHECK-NEXT:  %1 = shape.shape_of %0 : tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}> -> tensor<2xindex>
//CHECK-NEXT:  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>
//CHECK-NEXT:  %3 = mhlo.add %0, %2 : tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>
//CHECK-NEXT:  return %3 : tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>

func.func @registered_shape_infer(%arg0 : tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}) -> tensor<?xi64> {
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "byteir.non_zero"} : (tensor<?x4xf32>) -> tensor<?xi64>
  return %0 : tensor<?xi64>
}

//CHECK-LABEL: func.func @registered_shape_infer(%arg0: tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}> {byteir.bounded_shape = [8, 4]}) -> tensor<?xi64, {byteir.bounded_shape = [32]}> {
//CHECK-NEXT: %0 = "mhlo.custom_call"(%arg0) {call_target_name = "byteir.non_zero"} : (tensor<?x4xf32, {byteir.bounded_shape = [8, 4]}>) -> tensor<?xi64, {byteir.bounded_shape = [32]}>
//CHECK-NEXT: return %0 : tensor<?xi64, {byteir.bounded_shape = [32]}>

func.func @tf_where(%arg0 : tensor<1xi1>) -> tensor<?x1xi64> {
  %0 = "mhlo.custom_call"(%arg0) { call_target_name = "tf.Where" } : (tensor<1xi1>) -> tensor<?x1xi64>
  return %0 : tensor<?x1xi64>
}

//CHECK-LABEL: func.func @tf_where(%arg0: tensor<1xi1>) -> tensor<?x1xi64, {byteir.bounded_shape = [1, 1]}> {
//CHECK-NEXT: %0 = "mhlo.custom_call"(%arg0) {call_target_name = "tf.Where"} : (tensor<1xi1>) -> tensor<?x1xi64, {byteir.bounded_shape = [1, 1]}>
//CHECK-NEXT: return %0 : tensor<?x1xi64, {byteir.bounded_shape = [1, 1]}>

func.func @main_sub_0(%arg0: tensor<?x4xf32> {byteir.bounded_shape = [4, 4]}) -> tensor<?xf32> {
  %0 = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x4xf32>, tensor<f32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

//CHECK-LABEL: func.func @main_sub_0(%arg0: tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}> {byteir.bounded_shape = [4, 4]}) -> tensor<?xf32, {byteir.bounded_shape = [4]}> {
//CHECK-NEXT:  %0 = mhlo.constant dense<-0.000000e+00> : tensor<f32>
//CHECK-NEXT:  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, tensor<f32>) -> tensor<?xf32, {byteir.bounded_shape = [4]}>
//CHECK-NEXT:  return %1 : tensor<?xf32, {byteir.bounded_shape = [4]}>

func.func @concat(%arg0 : tensor<?x3xf32> {byteir.bounded_shape = [3, 3]}, %arg1 : tensor<?x3xf32> {byteir.bounded_shape = [3, 3]}) -> tensor <?x6xf32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x6xf32>
  return %0 : tensor<?x6xf32>
}
//CHECK-LABEL: func.func @concat(%arg0: tensor<?x3xf32, {byteir.bounded_shape = [3, 3]}> {byteir.bounded_shape = [3, 3]}, %arg1: tensor<?x3xf32, {byteir.bounded_shape = [3, 3]}> {byteir.bounded_shape = [3, 3]}) -> tensor<?x6xf32, {byteir.bounded_shape = [3, 6]}> {
//CHECK-NEXT:  %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<?x3xf32, {byteir.bounded_shape = [3, 3]}>, tensor<?x3xf32, {byteir.bounded_shape = [3, 3]}>) -> tensor<?x6xf32, {byteir.bounded_shape = [3, 6]}>

func.func @dynamic_reshape(%arg0 : tensor<?x1xi64> {byteir.bounded_shape = [100, 1]}) -> tensor <?xi64> {
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x1xi64>
  %1 = tensor.from_elements %0 : tensor<1xindex>
  %2 = shape.shape_of %arg0 : tensor<?x1xi64> -> tensor<2xindex>
  %3 = shape.num_elements %2 : tensor<2xindex> -> index
  %4 = mhlo.cstr_reshapable %3, %1 : index, tensor<1xindex>
  %5 = shape.assuming %4 -> (tensor<?xi64>) {
    %6 = mhlo.compute_reshape_shape %3, %1 : index, tensor<1xindex> -> tensor<1xindex>
    %7 = "mhlo.dynamic_reshape"(%arg0, %6) : (tensor<?x1xi64>, tensor<1xindex>) -> tensor<?xi64>
    shape.assuming_yield %7 : tensor<?xi64>
  }
  return %5 : tensor<?xi64>
}

// CHECK-LABEL: func.func @dynamic_reshape(%arg0: tensor<?x1xi64, {byteir.bounded_shape = [100, 1]}> {byteir.bounded_shape = [100, 1]}) -> tensor<?xi64, {byteir.bounded_shape = [100]}> {
// CHECK: %7 = "mhlo.dynamic_reshape"(%arg0, %6) : (tensor<?x1xi64, {byteir.bounded_shape = [100, 1]}>, tensor<1xindex>) -> tensor<?xi64, {byteir.bounded_shape = [100]}>
// CHECK: return %5 : tensor<?xi64, {byteir.bounded_shape = [100]}>

func.func @dynamic_broadcast_in_dim(%arg0 : tensor<?x32xf32> {byteir.bounded_shape = [32, 32]}) -> tensor <1x?x30x32xf32> {
  %c30 = arith.constant 30 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x32xf32>
  %1 = tensor.from_elements %c1, %0, %c30, %c32 : tensor<4xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %1) {broadcast_dimensions = dense<[1, 3]> : tensor<2xi64>} : (tensor<?x32xf32>, tensor<4xindex>) -> tensor<1x?x30x32xf32>
  return %2 : tensor<1x?x30x32xf32>
}
//CHECK-LABEL: func.func @dynamic_broadcast_in_dim(%arg0: tensor<?x32xf32, {byteir.bounded_shape = [32, 32]}> {byteir.bounded_shape = [32, 32]}) -> tensor<1x?x30x32xf32, {byteir.bounded_shape = [1, 32, 30, 32]}> {
//CHECK: %2 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %1) {broadcast_dimensions = dense<[1, 3]> : tensor<2xi64>} : (tensor<?x32xf32, {byteir.bounded_shape = [32, 32]}>, tensor<4xindex>) -> tensor<1x?x30x32xf32, {byteir.bounded_shape = [1, 32, 30, 32]}>

func.func @torch_index_select(%arg0: tensor<10x128xf16>, %arg1: tensor<?xi32> {byteir.bounded_shape = [10]}) -> tensor<?x128xf16> {
    %6 = "mhlo.torch_index_select"(%arg0, %arg1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<10x128xf16>, tensor<?xi32>) -> tensor<?x128xf16>
    return %6 : tensor<?x128xf16>
}

// CHECK-LABEL: func.func @torch_index_select(%arg0: tensor<10x128xf16>, %arg1: tensor<?xi32, {byteir.bounded_shape = [10]}> {byteir.bounded_shape = [10]}) -> tensor<?x128xf16, {byteir.bounded_shape = [10, 128]}>
// CHECK-NEXT:  %0 = "mhlo.torch_index_select"(%arg0, %arg1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<10x128xf16>, tensor<?xi32, {byteir.bounded_shape = [10]}>) -> tensor<?x128xf16, {byteir.bounded_shape = [10, 128]}>
// CHECK-NEXT:  return %0 : tensor<?x128xf16, {byteir.bounded_shape = [10, 128]}>
