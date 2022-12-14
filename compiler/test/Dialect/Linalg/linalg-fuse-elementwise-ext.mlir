// RUN: byteir-opt %s -linalg-fuse-elementwise-ext="shared-input" -cse -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @two_outputs
func.func @two_outputs(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       
      %4 = arith.addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%3, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
      // CHECK-NOT: linalg.yield
      // CHECK: [[T2:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[ARG2]]
      // CHECK: linalg.yield [[T1]], [[T2]]   
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %3, %4 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @two_use_two_outputs
func.func @two_use_two_outputs(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %4 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%0, %0 : tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32):
    %2 = arith.addf %in, %in_1 : f32
    %3 = arith.mulf %2, %in_2 : f32
    linalg.yield %2, %3 : f32, f32
  } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK: linalg.generic {
  %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%1#0, %1#1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%4 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):  
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
      // CHECK-NOT: linalg.yield
      // CHECK: [[T2:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[ARG2]]
      // CHECK: [[T3:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[T2]]
      // CHECK: linalg.yield [[T1]], [[T3]]    
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %1#0, %6 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @one_use_0_two_outputs
func.func @one_use_0_two_outputs(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>, %arg3 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %4 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%0, %0 : tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32):
    %2 = arith.addf %in, %in_1 : f32
    %3 = arith.mulf %2, %in_2 : f32
    linalg.yield %2, %3 : f32, f32
  } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK: linalg.generic {
  %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%1#0, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%4 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):  
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
      // CHECK-NOT: linalg.yield
      // CHECK: [[T2:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[ARG2]]
      // CHECK: linalg.yield [[T1]], [[T2]]    
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %1#0, %6 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @one_use_1_two_outputs
func.func @one_use_1_two_outputs(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>, %arg3 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %4 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%0, %0 : tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32):
    %2 = arith.addf %in, %in_1 : f32
    %3 = arith.mulf %2, %in_2 : f32
    linalg.yield %2, %3 : f32, f32
  } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK: linalg.generic {
  %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%1#1, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%4 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG3:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):  
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
      // CHECK-NOT: linalg.yield
      // CHECK: [[T2:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[ARG2]]
      // CHECK: [[T3:%[a-zA-Z0-9_]*]] = arith.mulf [[T2]], [[ARG3]]
      // CHECK: linalg.yield [[T1]], [[T3]]    
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %1#0, %6 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @two_consumers
func.func @two_consumers(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>, %arg3 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):       
      %4 = arith.addf %arg4, %arg5 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%3, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%3, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG3:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
      // CHECK-NOT: linalg.yield
      // CHECK: [[T2:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[ARG2]]
      // CHECK: [[T3:%[a-zA-Z0-9_]*]] = arith.mulf [[T1]], [[ARG3]]
      // CHECK: linalg.yield [[T2]], [[T3]]   
      %6 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %6 : f32
    } -> tensor<?x?xf32>
  return %4, %5 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @shared_input
func.func @shared_input(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       
      %5 = arith.addf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
// CHECK: linalg.generic {
  %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):
    // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
    // CHECK-NOT: linalg.yield
    // CHECK: [[T2:%[a-zA-Z0-9_]*]] = arith.mulf [[ARG0]], [[ARG2]]
    // CHECK: linalg.yield [[T1]], [[T2]]  
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %4, %5 : tensor<?x?xf32>, tensor<?x?xf32>
}