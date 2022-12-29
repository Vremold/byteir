// RUN: byteir-opt %s --test-transform-dialect-interpreter --canonicalize-ext --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @fuse_unary
func.func @fuse_unary(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK:    scf.for
  //     CHECK:       linalg.elemwise_unary
  //     CHECK:       linalg.elemwise_binary
  //     CHECK: return %[[RES]]
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg1
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [32, 32], tile_interchange = [0, 1]}
}

// -----

// CHECK-LABEL: func.func @fuse_unary
func.func @fuse_unary(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  //     CHECK: %[[PARTIAL_RES:.*]] = scf.for
  //     CHECK:     scf.for
  //     CHECK:       linalg.elemwise_unary
  //     CHECK:       linalg.elemwise_binary
  //     CHECK: %[[RES:.*]] = scf.for {{.*}}%[[PARTIAL_RES]]
  //     CHECK:     scf.for
  //     CHECK:       linalg.elemwise_unary
  //     CHECK:       linalg.elemwise_binary
  //     CHECK: return %[[RES]]
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg1
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [32, 32], tile_interchange = [0, 1]}
  %loop = transform.cast %loops#0 : !pdl.operation to !transform.op<"scf.for">
  transform.loop.peel %loop : (!transform.op<"scf.for">) -> !pdl.operation
}

// -----

// CHECK-LABEL: func.func @interchange_reduction
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<12x7x25xf32>)
func.func @interchange_reduction(%input: tensor<12x7x25xf32>) -> tensor<12x25xf32> {
  %five = arith.constant 5.0 : f32
  %init = tensor.empty() : tensor<12x25xf32>

//   CHECK-DAG: %[[INIT:.+]] = tensor.empty()
//   CHECK-DAG: %[[C5:.+]] = arith.constant 5 : index
//   CHECK-DAG: %[[C7:.+]] = arith.constant 7 : index
//   CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
//       CHECK: %[[RES:.*]] = scf.for %[[IV0:.+]] = %{{.+}} to %{{.+}} step %[[C5]] iter_args(%[[FOR_ARG0:.+]] = %[[INIT]])
//       CHECK:   scf.for %[[IV1:.+]] = %{{.+}} to %{{.+}} step %[[C7]] iter_args(%[[FOR_ARG1:.+]] = %[[FOR_ARG0]])
//       CHECK:     %[[OUT_SLICE0:.+]] = tensor.extract_slice %[[INPUT]][%[[IV0]], 0, %[[IV1]]]
//       CHECK:     %[[OUT_SLICE1:.+]] = tensor.extract_slice %[[FOR_ARG1]][%[[IV0]], %[[IV1]]]
//       CHECK:     %[[FILL:.+]] = linalg.fill {{.+}} outs(%[[OUT_SLICE1]] : tensor<?x?xf32>)
//
// Extra 4 constant is introduced, discard it.
//       CHECK:     scf.for %[[IV2:.+]] = %{{.+}} to %{{.+}} step %[[C4]] iter_args(%[[FOR_ARG2:.+]] = %[[FILL]])
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[OUT_SLICE0]]
//       CHECK:       %[[OUT_SLICE2:.+]] = tensor.extract_slice %[[FOR_ARG2]][0, 0]
//       CHECK:       linalg.generic {{.+}} ins(%[[IN_SLICE]] : tensor<?x?x?xf32>) outs(%[[OUT_SLICE2]] : tensor<?x?xf32>)
//       CHECK: return %[[RES]]

  %fill = linalg.fill ins(%five : f32) outs(%init : tensor<12x25xf32>) -> tensor<12x25xf32>
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

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [5, 0, 7], tile_interchange = [0, 2, 1]}
  %2, %loops_2 = transform.structured.tile %1 [0, 4]
}

// -----

// CHECK-LABEL: func.func @fuse_unary_softmax
func.func @fuse_unary_softmax(%arg0: tensor<1024x64xf32>, %arg1: tensor<1024x64xf32>) -> tensor<1024x64xf32> {

  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK:   linalg.elemwise_unary
  //     CHECK:   linalg_ext.softmax
  //     CHECK: } {__byteir_parallel__}
  //     CHECK: return %[[RES]]
  %0 = tensor.empty() : tensor<1024x64xf32>
  %1 = tensor.empty() : tensor<1024xf32>
  %2 = tensor.empty() : tensor<1024xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %4 = linalg.elemwise_unary ins(%arg0 : tensor<1024x64xf32>)
                             outs(%arg1: tensor<1024x64xf32>) -> tensor<1024x64xf32>
  %5:4 = linalg_ext.softmax
    dimension(1) 
    ins(%4 : tensor<1024x64xf32>) outs(%0, %1, %2, %3 : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  return %5#0 : tensor<1024x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.softmax"]} in %arg1
  %1, %loops = transform.structured.fuse_ext %0 {tile_sizes = [4], tile_interchange = [0]}
  transform.structured.tile_label %1
}


// -----

// CHECK-LABEL: func.func @fuse_matmul_softmax
func.func @fuse_matmul_softmax(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<1024x64xf32> {

  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK:   linalg.matmul
  //     CHECK:   linalg_ext.softmax
  //     CHECK: } {__byteir_parallel__}
  //     CHECK: return %[[RES]]
  %0 = tensor.empty() : tensor<1024x64xf32>
  %2 = tensor.empty() : tensor<1024x64xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %4 = tensor.empty() : tensor<1024xf32>
  %5 = tensor.empty() : tensor<1024xf32>
  %1 = linalg.matmul  ins(%arg0, %arg1: tensor<1024x32xf32>, tensor<32x64xf32>)
                     outs(%0: tensor<1024x64xf32>)
    -> tensor<1024x64xf32>

  %6:4 = linalg_ext.softmax
    dimension(1) 
    ins(%1 : tensor<1024x64xf32>) outs(%2, %3, %4, %5 : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
    
  return %6#0 : tensor<1024x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.softmax"]} in %arg1
  %1, %loop = transform.structured.fuse_ext %0 {tile_sizes = [4], tile_interchange = [0]}
  transform.structured.tile_label %1
}

// -----

// CHECK-LABEL: func.func @fuse_matmul_matmul
func.func @fuse_matmul_matmul(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x512xf32>, %arg2: tensor<512x32xf32>) -> tensor<1024x32xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.matmul
// CHECK:     linalg.matmul
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   scf.yield
// CHECK: } {__byteir_parallel__}
  %0 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x32xf32>
  %1 = linalg.matmul  ins(%arg0, %arg1: tensor<1024x32xf32>, tensor<32x512xf32>)
                     outs(%0: tensor<1024x512xf32>)
    -> tensor<1024x512xf32>

  %3 = linalg.matmul {__root__} ins(%1, %arg2: tensor<1024x512xf32>, tensor<512x32xf32>)
                     outs(%2: tensor<1024x32xf32>)
    -> tensor<1024x32xf32>
  return %3 : tensor<1024x32xf32>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [4, 0, 8], tile_interchange = [0, 1, 2]}
  transform.structured.tile_label %1
}

// -----

// CHECK-LABEL: func.func @fuse_dot_attention
func.func @fuse_dot_attention(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x512xf32>, %arg2: tensor<512x32xf32>) -> tensor<1024x32xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.matmul
// CHECK:     linalg_ext.softmax
// CHECK:     linalg_ext.diag
// CHECK:     linalg.matmul
// CHECK:     linalg.matmul
// CHECK:     scf.yield
// CHECK:   } {__byteir_parallel__}
// CHECK:   scf.yield
// CHECK: }
  %0 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x32xf32>
  %3 = tensor.empty() : tensor<1024x512xf32>
  %4 = tensor.empty() : tensor<1024xf32>
  %5 = tensor.empty() : tensor<1024xf32>
  %6 = tensor.empty() : tensor<1024xf32>
  %1 = linalg.matmul  ins(%arg0, %arg1: tensor<1024x32xf32>, tensor<32x512xf32>)
                     outs(%0: tensor<1024x512xf32>)
    -> tensor<1024x512xf32>

  %7:4 = linalg_ext.softmax dimension(1) 
    ins(%1 : tensor<1024x512xf32>) outs(%3, %4, %5, %6 : tensor<1024x512xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x512xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>  

  %8 = linalg.matmul {__root__} ins(%7#0, %arg2: tensor<1024x512xf32>, tensor<512x32xf32>)
                     outs(%2: tensor<1024x32xf32>)
    -> tensor<1024x32xf32>
  return %8: tensor<1024x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 0, 8], tile_interchange = [2, 1, 0]}
  transform.structured.tile_label %1 
}

// -----

// CHECK-LABEL: func.func @fuse_2_matmul_2_output
func.func @fuse_2_matmul_2_output(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x512xf32>, %arg2: tensor<512x32xf32>) -> (tensor<1024x32xf32>, tensor<1024x512xf32>) {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.matmul
// CHECK:     linalg.matmul
// CHECK:     scf.yield
// CHECK:   } {__byteir_parallel__}
// CHECK:   scf.yield
// CHECK: }
  %0 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x32xf32>
  %3 = tensor.empty() : tensor<1024x512xf32>
  %4 = tensor.empty() : tensor<1024xf32>
  %5 = tensor.empty() : tensor<1024xf32>
  %6 = tensor.empty() : tensor<1024xf32>
  %1 = linalg.matmul  ins(%arg0, %arg1: tensor<1024x32xf32>, tensor<32x512xf32>)
                     outs(%0: tensor<1024x512xf32>)
    -> tensor<1024x512xf32>

  %8 = linalg.matmul {__root__} ins(%1, %arg2: tensor<1024x512xf32>, tensor<512x32xf32>)
                     outs(%2: tensor<1024x32xf32>)
    -> tensor<1024x32xf32>
  return %8, %1: tensor<1024x32xf32>, tensor<1024x512xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 0, 8], tile_interchange = [2, 1, 0]}
  transform.structured.tile_label %1 
}

// -----

// CHECK-LABEL: func.func @fuse_2_matmul_add
func.func @fuse_2_matmul_add(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x512xf32>, %arg2: tensor<1024x32xf32>, %arg3: tensor<32x512xf32>) -> tensor<1024x512xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.matmul
// CHECK:     linalg.matmul
// CHECK:     linalg.elemwise_binary
// CHECK:     scf.yield
// CHECK:   } {__byteir_parallel__}
// CHECK:   scf.yield
// CHECK: } {__byteir_parallel__}
  %0 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x512xf32>
  %4 = tensor.empty() : tensor<1024x512xf32>
  %1 = linalg.matmul  ins(%arg0, %arg1: tensor<1024x32xf32>, tensor<32x512xf32>)
                     outs(%0: tensor<1024x512xf32>)
    -> tensor<1024x512xf32>
  %3 = linalg.matmul  ins(%arg2, %arg3: tensor<1024x32xf32>, tensor<32x512xf32>)
                     outs(%2: tensor<1024x512xf32>)
    -> tensor<1024x512xf32>


  %5 = linalg.elemwise_binary {__root__} ins(%1, %3 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%4: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  
  return %5: tensor<1024x512xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [4, 8], tile_interchange = [1, 0]}
  transform.structured.tile_label %1 
}

// -----

// CHECK-LABEL: func.func @fuse_fork_add
func.func @fuse_fork_add(%arg0: tensor<1024x512xf32>, %arg1: tensor<1024x512xf32>, %arg2: tensor<1024x512xf32>) -> tensor<1024x512xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.elemwise_binary
// CHECK:     linalg.elemwise_binary
// CHECK:     linalg.elemwise_binary
// CHECK:     scf.yield
// CHECK:   } {__byteir_parallel__}
// CHECK:   scf.yield
// CHECK: } {__byteir_parallel__}
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x512xf32>
  %3 = linalg.elemwise_binary ins(%arg0, %arg1 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%0: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %4 = linalg.elemwise_binary ins(%arg1, %arg2 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%1: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %5 = linalg.elemwise_binary {__root__} ins(%3, %4 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%2: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  return %5: tensor<1024x512xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [4, 8], tile_interchange = [1, 0]}
  transform.structured.tile_label %1 
}

// -----

// CHECK-LABEL: func.func @fuse_fork_map
func.func @fuse_fork_map(%arg0: tensor<1024x512xf32>, %arg1: tensor<1024x512xf32>, %arg2: tensor<1024x512xf32>) -> tensor<1024x512xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.map
// CHECK:     linalg.map
// CHECK:     linalg.map
// CHECK:     scf.yield
// CHECK:   } {__byteir_parallel__}
// CHECK:   scf.yield
// CHECK: } {__byteir_parallel__}
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x512xf32>
  %3 = linalg.map
      ins(%arg0, %arg1: tensor<1024x512xf32>, tensor<1024x512xf32>)
      outs(%0:tensor<1024x512xf32>)
      (%lhs_elem: f32, %rhs_elem: f32) {
        %6 = arith.addf %lhs_elem, %rhs_elem: f32
        linalg.yield %6: f32
      }
  %4 = linalg.map
      ins(%arg1, %arg2: tensor<1024x512xf32>, tensor<1024x512xf32>)
      outs(%1:tensor<1024x512xf32>)
      (%lhs_elem: f32, %rhs_elem: f32) {
        %6 = arith.mulf %lhs_elem, %rhs_elem: f32
        linalg.yield %6: f32
      }
  %5 = linalg.map
      ins(%3, %4: tensor<1024x512xf32>, tensor<1024x512xf32>)
      outs(%2:tensor<1024x512xf32>)  {__root__} 
      (%lhs_elem: f32, %rhs_elem: f32) {
        %6 = arith.addf %lhs_elem, %rhs_elem: f32
        linalg.yield %6: f32
      }
  return %5: tensor<1024x512xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [4, 8], tile_interchange = [1, 0]}
  transform.structured.tile_label %1 
}

// -----

// CHECK-LABEL: func.func @fuse_2_add_sharing_add
func.func @fuse_2_add_sharing_add(%arg0: tensor<1024x512xf32>, %arg1: tensor<1024x512xf32>, %arg2: tensor<1024x512xf32>, %arg3: tensor<1024x512xf32>) -> (tensor<1024x512xf32>,tensor<1024x512xf32>) {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.elemwise_binary
// CHECK:     linalg.elemwise_binary {__root__}
// CHECK:     scf.yield
// CHECK:   } {__byteir_parallel__}
// CHECK:   scf.yield
// CHECK: } {__byteir_parallel__}
// CHECK: linalg.elemwise_binary
// CHECK: linalg.elemwise_binary {__other__} 
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x512xf32>
  %3 = linalg.elemwise_binary ins(%arg0, %arg1 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%0: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %4 = linalg.elemwise_binary ins(%arg2, %3 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%1: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %6 = linalg.elemwise_binary {__other__} ins(%arg2, %4 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%1: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %5 = linalg.elemwise_binary {__root__} ins(%arg3, %3 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%2: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  return %6, %5: tensor<1024x512xf32>, tensor<1024x512xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 8], tile_interchange = [1, 0]}
  transform.structured.tile_label %1 
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @max_pool_generic
func.func @max_pool_generic(%arg0: tensor<4x126x126x16xf32>) -> tensor<4x63x63x16xf32> {
// CHECK: scf.for
// CHECK:   linalg.generic
// CHEKC-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK:     arith.maxf
// CHECK:     linalg.yield
// CHECK: scf.yield
  %cst = arith.constant dense<0xFC00> : tensor<4x63x63x16xf32>
  %0 = tensor.empty() : tensor<2x2xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %0 : tensor<4x126x126x16xf32>, tensor<2x2xf32>) outs(%cst : tensor<4x63x63x16xf32>) attrs =  {__tiling_0} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %5 = arith.maxf %out, %in : f32
    linalg.yield %5 : f32
  } -> tensor<4x63x63x16xf32>
  return %1 : tensor<4x63x63x16xf32>
}
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__tiling_0} in %arg0
  %transformed, %loops = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [1]}
}


