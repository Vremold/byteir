// RUN: byteir-opt %s --test-transform-dialect-interpreter --canonicalize-ext --split-input-file | FileCheck %s

// this is called dev, since it is not perfect yet.

// CHECK-LABEL: func.func @fuse_2_add_sharing_input
func.func @fuse_2_add_sharing_input(%arg0: tensor<1024x512xf32>, %arg1: tensor<1024x512xf32>, %arg2: tensor<1024x512xf32>) -> (tensor<1024x512xf32>,tensor<1024x512xf32>) {
// CHECK: linalg.elemwise_binary {__other__}
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.elemwise_binary {__root__}
// CHECK:     scf.yield
// CHECK:   } {__byteir_parallel__}
// CHECK:   scf.yield
// CHECK: } {__byteir_parallel__}
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x512xf32>
  %3 = linalg.elemwise_binary {__other__} ins(%arg0, %arg1 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%0: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %4 = linalg.elemwise_binary {__root__} ins(%arg1, %arg2 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%1: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  return %3, %4: tensor<1024x512xf32>, tensor<1024x512xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 8], tile_interchange = [1, 0]}
  transform.structured.tile_label %1 
}

// -----

// CHECK-LABEL: func.func @fuse_elementwise_dynamic_of_intermediate
func.func @fuse_elementwise_dynamic_of_intermediate(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[T0:.*]] = linalg.elemwise_unary
  // CHECK-DAG: %[[D1:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
  // CHECK-DAG: %[[D0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
  // CHECK: %[[RES:.*]] = scf.for
  // CHECK:    scf.for
  // CHECK:       linalg.elemwise_unary
  // CHECK:       linalg.elemwise_binary
  // CHECK: return %[[RES]]
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg1
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [32, 32], tile_interchange = [0, 1]}
}
