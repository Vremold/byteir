// RUN: byteir-opt -resolve-shape-constraint %s | FileCheck %s

// CHECK-LABEL: @dynamic_partition_constraint
func @dynamic_partition_constraint(%arg0: tensor<4x4xf32>, %arg1: tensor<4xi32>) -> tensor<?x4xf32> {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %0 = mhlo.constant dense<[[-0.916170597, -0.884184718, 1.60242105, -1.19678485], [0.33643803, -0.431175768, 1.71861267, 0.126368985], [-1.07191086, -1.00517535, -0.666032254, 0.776807785], [1.53380013, 0.83925873, -0.24277249, 1.53341103]]> : tensor<4x4xf32>
  %1 = mhlo.constant dense<[2.63629675, 2.68127704, 2.14741468, -1.6519475]> : tensor<4xf32>
  %2 = mhlo.constant dense<[[0.473984271, 0.173930168, 0.465745121, 1.14254773], [-0.384602815, -0.673360229, 1.13109767, 0.761463344], [-0.171464354, -0.908823907, 1.19337058, -1.78143835], [1.40376866, -0.529214859, -1.9030931, 1.25083804]]> : tensor<4x4xf32>
  %3 = mhlo.constant dense<[0.478572756, 0.458867788, -1.44476604, 0.189240679]> : tensor<4xf32>
  %4 = mhlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %5:2 = "mhlo.custom_call"(%arg0, %arg1) {byteir_attrs = {num_partitions = 2 : i64}, call_target_name = "byteir.dynamic_partition", has_side_effect = false} : (tensor<4x4xf32>, tensor<4xi32>) -> (tensor<?x4xf32>, tensor<?x4xf32>)
  %6 = tensor.dim %5#0, %c0 : tensor<?x4xf32>
  %7 = tensor.dim %5#1, %c0 : tensor<?x4xf32>
  %8 = shape.add %6, %7 : index, index -> index
  "shape_ext.meet"(%8, %c4) : (index, index) -> ()
  "shape_ext.tie"(%5#1, %7) : (tensor<?x4xf32>, index) -> ()
  "shape_ext.tie"(%5#0, %6) : (tensor<?x4xf32>, index) -> ()
  %9 = "mhlo.dot"(%5#0, %0) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  "shape_ext.tie"(%9, %6) : (tensor<?x4xf32>, index) -> ()
  %10 = tensor.from_elements %6, %c4 : tensor<2xindex>
  %11 = "mhlo.dynamic_broadcast_in_dim"(%1, %10) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  "shape_ext.tie"(%11, %6) : (tensor<?x4xf32>, index) -> ()
  %12 = mhlo.add %9, %11 : tensor<?x4xf32>
  "shape_ext.tie"(%12, %6) : (tensor<?x4xf32>, index) -> ()
  %13 = "mhlo.dot"(%5#1, %2) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  "shape_ext.tie"(%13, %7) : (tensor<?x4xf32>, index) -> ()
  %14 = tensor.from_elements %7, %c4 : tensor<2xindex>
  %15 = "mhlo.dynamic_broadcast_in_dim"(%3, %14) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  "shape_ext.tie"(%15, %7) : (tensor<?x4xf32>, index) -> ()
  %16 = mhlo.add %13, %15 : tensor<?x4xf32>
  "shape_ext.tie"(%16, %7) : (tensor<?x4xf32>, index) -> ()
  %17:2 = "mhlo.custom_call"(%4, %arg1) {byteir_attrs = {num_partitions = 2 : i64}, call_target_name = "byteir.dynamic_partition", has_side_effect = false} : (tensor<4xi32>, tensor<4xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %18 = tensor.dim %17#0, %c0 : tensor<?xi32>
  %19 = tensor.dim %17#1, %c0 : tensor<?xi32>
  %20 = shape.add %18, %19 : index, index -> index
  "shape_ext.meet"(%20, %c4) : (index, index) -> ()
  "shape_ext.tie"(%17#1, %19) : (tensor<?xi32>, index) -> ()
  "shape_ext.tie"(%17#0, %18) : (tensor<?xi32>, index) -> ()
  // CHECK: "byteir.dynamic_stitch"{{.*}}: (tensor<?xi32>, tensor<?xi32>, tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<4x4xf32>
  %21 = "mhlo.custom_call"(%17#0, %17#1, %12, %16) {call_target_name = "byteir.dynamic_stitch", has_side_effect = false} : (tensor<?xi32>, tensor<?xi32>, tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  // CHECK-NOT: "shape_ext.tie"
  "shape_ext.tie"(%21, %20) : (tensor<?x4xf32>, index) -> ()
  return %21 : tensor<?x4xf32>
}

func @einsum_shape_constraint(%arg0: tensor<?x2x2xf32>, %arg1: tensor<2x2x3xf32>, %arg2: tensor<2x2x2xf32>) -> tensor<2x2x2x3xf32> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = tensor.dim %arg0, %c0 : tensor<?x2x2xf32>
    "shape_ext.tie"(%arg0, %0) : (tensor<?x2x2xf32>, index) -> ()
    %1 = "mhlo.einsum"(%arg1, %arg0) {einsum_config = "edc,bqe->dcbq"} : (tensor<2x2x3xf32>, tensor<?x2x2xf32>) -> tensor<2x3x?x2xf32>
    "shape_ext.meet"(%c2, %c2) : (index, index) -> ()
    "shape_ext.meet"(%c3, %c3) : (index, index) -> ()
    %2 = tensor.dim %1, %c2 : tensor<2x3x?x2xf32>
    "shape_ext.meet"(%0, %2) : (index, index) -> ()
    "shape_ext.meet"(%c2, %c2) : (index, index) -> ()
    "shape_ext.meet"(%c2, %c2) : (index, index) -> ()
    "shape_ext.tie"(%1, %2) : (tensor<2x3x?x2xf32>, index) -> ()
    %3 = "mhlo.einsum"(%1, %arg2) {einsum_config = "dcbq,btd->bqtc"} : (tensor<2x3x?x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2x3xf32>
    "shape_ext.meet"(%c2, %2) : (index, index) -> ()
    "shape_ext.meet"(%c2, %c2) : (index, index) -> ()
    "shape_ext.meet"(%c2, %c2) : (index, index) -> ()
    "shape_ext.meet"(%c2, %c2) : (index, index) -> ()
    "shape_ext.meet"(%c3, %c3) : (index, index) -> ()
    "shape_ext.meet"(%c2, %c2) : (index, index) -> ()
    return %3 : tensor<2x2x2x3xf32>
}
// CHECK-LABEL: @einsum_shape_constraint(%arg0: tensor<2x2x2xf32>
// CHECK-NEXT: %0 = "mhlo.einsum"(%arg1, %arg0) {einsum_config = "edc,bqe->dcbq"} : (tensor<2x2x3xf32>, tensor<2x2x2xf32>) -> tensor<2x3x2x2xf32>
// CHECK-NEXT: %1 = "mhlo.einsum"(%0, %arg2) {einsum_config = "dcbq,btd->bqtc"} : (tensor<2x3x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2x3xf32>
// CHECK-NEXT: return %1 : tensor<2x2x2x3xf32>

