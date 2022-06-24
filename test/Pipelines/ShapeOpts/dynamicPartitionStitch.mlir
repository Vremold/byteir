// RUN: byteir-opt %s -shape-opt | FileCheck %s

func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4xi32>) -> tensor<?x4xf32> {
  %0 = shape.const_shape [4] : tensor<1xindex>
  %1 = mhlo.constant dense<[[-0.705530286, 0.87041223, 0.972314774, -0.0584422052], [-1.43617868, 6.772900e-01, 0.880922436, 0.56821847], [0.57929492, 0.470399499, -1.0485183, -1.27004325], [-0.32425791, 1.88410747, 0.220974803, -0.238485783]]> : tensor<4x4xf32>
  %2 = mhlo.constant dense<[0.553816557, -0.920699775, 0.418103188, -0.261674613]> : tensor<4xf32>
  %3 = mhlo.constant dense<[[-0.916170597, -0.884184718, 1.60242105, -1.19678485], [0.33643803, -0.431175768, 1.71861267, 0.126368985], [-1.07191086, -1.00517535, -0.666032254, 0.776807785], [1.53380013, 0.83925873, -0.24277249, 1.53341103]]> : tensor<4x4xf32>
  %4 = mhlo.constant dense<[2.63629675, 2.68127704, 2.14741468, -1.6519475]> : tensor<4xf32>
  %5 = mhlo.constant dense<[[0.473984271, 0.173930168, 0.465745121, 1.14254773], [-0.384602815, -0.673360229, 1.13109767, 0.761463344], [-0.171464354, -0.908823907, 1.19337058, -1.78143835], [1.40376866, -0.529214859, -1.9030931, 1.25083804]]> : tensor<4x4xf32>
  %6 = mhlo.constant dense<[0.478572756, 0.458867788, -1.44476604, 0.189240679]> : tensor<4xf32>
  %7 = mhlo.constant dense<[[-1.87686706, 0.286330104, -0.044809185, -0.178677231], [-1.14233077, -0.446333855, -1.2957921, 0.446576297], [0.985618114, 0.699275255, 0.609199941, -0.726590812], [0.0366623849, -0.640842735, -1.72003555, -0.383472085]]> : tensor<4x4xf32>
  %8 = mhlo.constant dense<[1.56364501, -0.948736965, 0.0843383893, 0.502355933]> : tensor<4xf32>
  %9 = mhlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %10 = "mhlo.dot"(%arg0, %1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %11 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<4x4xf32>
  %12 = mhlo.add %10, %11 : tensor<4x4xf32>
  %13:2 = "mhlo.custom_call"(%12, %arg1) {byteir_attrs = {num_partitions = 2 : i64}, call_target_name = "byteir.dynamic_partition", has_side_effect = false} : (tensor<4x4xf32>, tensor<4xi32>) -> (tensor<?x4xf32>, tensor<?x4xf32>)
  %14 = "mhlo.dot"(%13#0, %3) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %15 = shape.shape_of %14 : tensor<?x4xf32> -> tensor<2xindex>
  %18 = "mhlo.dynamic_broadcast_in_dim"(%4, %15) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %19 = mhlo.add %14, %18 : tensor<?x4xf32>
  %20 = "mhlo.dot"(%13#1, %5) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %21 = shape.shape_of %20 : tensor<?x4xf32> -> tensor<2xindex>
  %24 = "mhlo.dynamic_broadcast_in_dim"(%6, %21) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %25 = mhlo.add %20, %24 : tensor<?x4xf32>
  %26:2 = "mhlo.custom_call"(%9, %arg1) {byteir_attrs = {num_partitions = 2 : i64}, call_target_name = "byteir.dynamic_partition", has_side_effect = false} : (tensor<4xi32>, tensor<4xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %27 = "mhlo.custom_call"(%26#0, %26#1, %19, %25) {call_target_name = "byteir.dynamic_stitch", has_side_effect = false} : (tensor<?xi32>, tensor<?xi32>, tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %28 = "mhlo.dot"(%27, %7) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %29 = shape.shape_of %28 : tensor<?x4xf32> -> tensor<2xindex>
  %32 = "mhlo.dynamic_broadcast_in_dim"(%8, %29) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %33 = mhlo.add %28, %32 : tensor<?x4xf32>
  return %33 : tensor<?x4xf32>
}
// CHECK-LABEL: @main
// CHECK: call @main_sub_2({{.*}}) : (tensor<?x4xf32>) -> tensor<?x4xf32>
// CHECK: call @main_sub_1({{.*}}) : (tensor<?x4xf32>) -> tensor<?x4xf32>
// CHECK: call @main_sub_0({{.*}}) : (tensor<?xi32>, tensor<?xi32>, tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<4x4xf32>

// CHECK-LABEL: func @main_sub_2(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
// CHECK-NEXT:   %c4 = arith.constant 4 : index
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %0 = mhlo.constant dense<[2.63629675, 2.68127704, 2.14741468, -1.6519475]> : tensor<4xf32>
// CHECK-NEXT:   %1 = mhlo.constant {{.*}} : tensor<4x4xf32>
// CHECK-NEXT:   %2 = tensor.dim %arg0, %c0 : tensor<?x4xf32>
// CHECK-NEXT:   %3 = "mhlo.dot"(%arg0, %1) {byteir.bounded_shape0 = [4, 4]} : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
// CHECK-NEXT:   %4 = tensor.from_elements %2, %c4 : tensor<2xindex>
// CHECK-NEXT:   %5 = "mhlo.dynamic_broadcast_in_dim"(%0, %4) {broadcast_dimensions = dense<1> : tensor<1xi64>, byteir.bounded_shape0 = [-1, 4]} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
// CHECK-NEXT:   %6 = mhlo.add %3, %5 {byteir.bounded_shape0 = [4, 4]} : tensor<?x4xf32>
// CHECK-NEXT:   return %6 : tensor<?x4xf32>
// CHECK-NEXT: }

// CHECK-LABEL: func @main_sub_1(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
// CHECK-NEXT:   %c4 = arith.constant 4 : index
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %0 = mhlo.constant dense<[0.478572756, 0.458867788, -1.44476604, 0.189240679]> : tensor<4xf32>
// CHECK-NEXT:   %1 = mhlo.constant {{.*}} : tensor<4x4xf32>
// CHECK-NEXT:   %2 = tensor.dim %arg0, %c0 : tensor<?x4xf32>
// CHECK-NEXT:   %3 = "mhlo.dot"(%arg0, %1) {byteir.bounded_shape0 = [4, 4]} : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
// CHECK-NEXT:   %4 = tensor.from_elements %2, %c4 : tensor<2xindex>
// CHECK-NEXT:   %5 = "mhlo.dynamic_broadcast_in_dim"(%0, %4) {broadcast_dimensions = dense<1> : tensor<1xi64>, byteir.bounded_shape0 = [-1, 4]} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
// CHECK-NEXT:   %6 = mhlo.add %3, %5 {byteir.bounded_shape0 = [4, 4]} : tensor<?x4xf32>
// CHECK-NEXT:   return %6 : tensor<?x4xf32>
// CHECK-NEXT: }

// CHECK-LABEL: func @main_sub_0(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>, %arg2: tensor<?x4xf32>, %arg3: tensor<?x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:   %0 = "mhlo.custom_call"(%arg0, %arg1, %arg2, %arg3) {call_target_name = "byteir.dynamic_stitch", has_side_effect = false} : (tensor<?xi32>, tensor<?xi32>, tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:   return %0 : tensor<4x4xf32>
// CHECK-NEXT: }