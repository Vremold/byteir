// RUN: byteir-opt %s -insert-shape-constraint -canonicalize -cse | FileCheck %s

// CHECK-LABEL: @dynamic_partition_constraint
func @dynamic_partition_constraint(%arg0: tensor<4x4xf32>, %arg1: tensor<4xi32>) -> tensor<?x4xf32> {
  %c0 = arith.constant 0 : index
  %0 = mhlo.constant dense<[[-0.916170597, -0.884184718, 1.60242105, -1.19678485], [0.33643803, -0.431175768, 1.71861267, 0.126368985], [-1.07191086, -1.00517535, -0.666032254, 0.776807785], [1.53380013, 0.83925873, -0.24277249, 1.53341103]]> : tensor<4x4xf32>
  %1 = mhlo.constant dense<[2.63629675, 2.68127704, 2.14741468, -1.6519475]> : tensor<4xf32>
  %2 = mhlo.constant dense<[[0.473984271, 0.173930168, 0.465745121, 1.14254773], [-0.384602815, -0.673360229, 1.13109767, 0.761463344], [-0.171464354, -0.908823907, 1.19337058, -1.78143835], [1.40376866, -0.529214859, -1.9030931, 1.25083804]]> : tensor<4x4xf32>
  %3 = mhlo.constant dense<[0.478572756, 0.458867788, -1.44476604, 0.189240679]> : tensor<4xf32>
  // CHECK-DAG: %[[C0:.*]] = mhlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %4 = mhlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  // CHECK-DAG: %[[V0:.*]]:2 = "mhlo.custom_call"(%arg0, %arg1) {byteir_attrs = {num_partitions = 2 : i64}, call_target_name = "byteir.dynamic_partition", has_side_effect = false} : (tensor<4x4xf32>, tensor<4xi32>) -> (tensor<?x4xf32>, tensor<?x4xf32>)
  %5:2 = "mhlo.custom_call"(%arg0, %arg1) {byteir_attrs = {num_partitions = 2 : i64}, call_target_name = "byteir.dynamic_partition", has_side_effect = false} : (tensor<4x4xf32>, tensor<4xi32>) -> (tensor<?x4xf32>, tensor<?x4xf32>)
  // CHECK-DAG: %[[V1:.*]] = tensor.dim %[[V0]]#0, %c0 : tensor<?x4xf32>
  // CHECK-DAG: %[[V2:.*]] = tensor.dim  %[[V0]]#1, %c0 : tensor<?x4xf32>
  // CHECK-DAG: %[[V3:.*]] = shape.add %[[V1]], %[[V2]] : index, index -> index
  // CHECK-DAG: "shape_ext.meet"(%[[V3]], %c4) : (index, index) -> ()
  %6 = tensor.dim %5#1, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%5#1, %6) : (tensor<?x4xf32>, index) -> ()
  %7 = tensor.dim %5#0, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%5#0, %7) : (tensor<?x4xf32>, index) -> ()
  %8 = "mhlo.dot"(%5#0, %0) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %9 = tensor.dim %8, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%8, %9) : (tensor<?x4xf32>, index) -> ()
  %10 = shape.shape_of %8 : tensor<?x4xf32> -> tensor<2xindex>
  %11 = "mhlo.dynamic_broadcast_in_dim"(%1, %10) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %12 = tensor.dim %11, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%11, %12) : (tensor<?x4xf32>, index) -> ()
  %13 = mhlo.add %8, %11 : tensor<?x4xf32>
  %14 = tensor.dim %13, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%13, %14) : (tensor<?x4xf32>, index) -> ()
  %15 = "mhlo.dot"(%5#1, %2) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %16 = tensor.dim %15, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%15, %16) : (tensor<?x4xf32>, index) -> ()
  %17 = shape.shape_of %15 : tensor<?x4xf32> -> tensor<2xindex>
  %18 = "mhlo.dynamic_broadcast_in_dim"(%3, %17) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %19 = tensor.dim %18, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%18, %19) : (tensor<?x4xf32>, index) -> ()
  %20 = mhlo.add %15, %18 : tensor<?x4xf32>
  %21 = tensor.dim %20, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%20, %21) : (tensor<?x4xf32>, index) -> ()
  // CHECK-DAG: %[[V4:.*]]:2 = "mhlo.custom_call"(%[[C0]], %arg1) {byteir_attrs = {num_partitions = 2 : i64}, call_target_name = "byteir.dynamic_partition", has_side_effect = false} : (tensor<4xi32>, tensor<4xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %22:2 = "mhlo.custom_call"(%4, %arg1) {byteir_attrs = {num_partitions = 2 : i64}, call_target_name = "byteir.dynamic_partition", has_side_effect = false} : (tensor<4xi32>, tensor<4xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  // CHECK-DAG: %[[V5:.*]] = tensor.dim %[[V4]]#0, %c0 : tensor<?xi32>
  // CHECK-DAG: %[[V6:.*]] = tensor.dim %[[V4]]#1, %c0 : tensor<?xi32>
  // CHECK-DAG: %[[V7:.*]] = shape.add %[[V5]], %[[V6]] : index, index -> index
  // CHECK-DAG: "shape_ext.meet"(%[[V7]], %c4) : (index, index) -> ()
  %23 = tensor.dim %22#1, %c0 : tensor<?xi32>
  "shape_ext.tie"(%22#1, %23) : (tensor<?xi32>, index) -> ()
  %24 = tensor.dim %22#0, %c0 : tensor<?xi32>
  "shape_ext.tie"(%22#0, %24) : (tensor<?xi32>, index) -> ()
  %25 = "mhlo.custom_call"(%22#0, %22#1, %13, %20) {call_target_name = "byteir.dynamic_stitch", has_side_effect = false} : (tensor<?xi32>, tensor<?xi32>, tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %26 = tensor.dim %25, %c0 : tensor<?x4xf32>
  "shape_ext.tie"(%25, %26) : (tensor<?x4xf32>, index) -> ()
  return %25 : tensor<?x4xf32>
}