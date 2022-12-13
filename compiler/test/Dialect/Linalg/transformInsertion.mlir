// RUN: byteir-opt %s --transform-insertion --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @elementwise
func.func @elementwise(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {__byteir_tiling_test__, device = "test"} {
// CHECK: linalg.elemwise_unary
// CHECK: linalg.elemwise_binary {__byteir_device_tiling__0}
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @max_pool
func.func @max_pool(%arg0: tensor<4x126x126x16xf32>) -> tensor<4x63x63x16xf32> attributes {__byteir_tiling_test__, device = "test"} {
    %cst = arith.constant dense<0xFF800000> : tensor<f32>
    %0 = tensor.empty() : tensor<2x2xf32>
    %1 = tensor.empty() : tensor<4x63x63x16xf32>
    %extracted = tensor.extract %cst[] : tensor<f32>
    %2 = linalg.fill ins(%extracted : f32) outs(%1 : tensor<4x63x63x16xf32>) -> tensor<4x63x63x16xf32>
    // CHECK:linalg.pooling_nhwc_max {__byteir_device_tiling__1
    %3 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %0 : tensor<4x126x126x16xf32>, tensor<2x2xf32>) outs(%2 : tensor<4x63x63x16xf32>) -> tensor<4x63x63x16xf32>
    return %3 : tensor<4x63x63x16xf32>
}

// CHECK-LABEL: func.func @generic
func.func @generic(%input: tensor<12x7x25xf32>) -> tensor<12x25xf32> attributes {__byteir_tiling_test__, device = "test"} {
  %five = arith.constant 5.0 : f32
  %init = tensor.empty() : tensor<12x25xf32>
  %fill = linalg.fill ins(%five : f32) outs(%init : tensor<12x25xf32>) -> tensor<12x25xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: attrs =  {__byteir_device_tiling__2} {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>],
    iterator_types = ["parallel", "reduction", "parallel"]
  } ins(%input : tensor<12x7x25xf32>) outs(%fill : tensor<12x25xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %2 = arith.addf %arg0, %arg1 : f32
    linalg.yield %2 : f32
  } -> tensor<12x25xf32>
  func.return %0 : tensor<12x25xf32>
}

// CHECK: transform.sequence failures(propagate) {
// CHECK: ^bb0(%arg0: !pdl.operation): 
// CHECK: %0 = transform.structured.match attributes {__byteir_device_tiling__0}
// CHECK: transform.structured.fuse_ext
// CHECK-SAME: {tile_interchange = [], tile_sizes = [1]}

// CHECK: transform.sequence failures(propagate) {
// CHECK: ^bb0(%arg0: !pdl.operation): 
// CHECK: %0 = transform.structured.match attributes {__byteir_device_tiling__1}
// CHECK: transform.structured.fuse_ext
// CHECK-SAME: {tile_interchange = [], tile_sizes = [1]}

// CHECK: transform.sequence failures(propagate) {
// CHECK: ^bb0(%arg0: !pdl.operation): 
// CHECK: transform.structured.match attributes {__byteir_device_tiling__2}
// CHECK: transform.structured.fuse_ext
// CHECK-SAME: {tile_interchange = [], tile_sizes = [1]}