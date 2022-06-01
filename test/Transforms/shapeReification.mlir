// RUN: byteir-opt %s -shape-reification -cse | FileCheck %s

func @several_ops(%arg0: tensor<?x2xf32>, %arg1: tensor<2x4xf32>, %arg2: tensor<4xf32>) -> (!shape.shape, !shape.shape, !shape.shape, !shape.shape) {                                                             
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x2xf32>, tensor<2x4xf32>) -> tensor<?x4xf32>               
  %1 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>                                         
  %2 = shape.value_as_shape %1 : tensor<2xindex> -> !shape.shape                                      
  %3 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>                                         
  %4 = shape.shape_of %3 : tensor<2xindex> -> tensor<1xindex>                                         
  %5 = shape.value_as_shape %4 : tensor<1xindex> -> !shape.shape                                      
  %6 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : 
(tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>                                                   
  %7 = shape.shape_of %6 : tensor<?x4xf32> -> tensor<2xindex>                                         
  %8 = shape.value_as_shape %7 : tensor<2xindex> -> !shape.shape                                      
  %9 = mhlo.add %0, %6 : tensor<?x4xf32>                                                              
  %10 = shape.shape_of %9 : tensor<?x4xf32> -> tensor<2xindex>                                        
  %11 = shape.value_as_shape %10 : tensor<2xindex> -> !shape.shape                                    
  return %2, %5, %8, %11 : !shape.shape, !shape.shape, !shape.shape, !shape.shape                     
}
// CHECK-LABEL: func @several_ops
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[C2:.+]] = shape.const_shape [2] : tensor<1xindex>
// CHECK-DAG:     %[[V0:.+]] = tensor.dim %arg0, %[[C0]] : tensor<?x2xf32>
// CHECK-DAG:     %[[V1:.+]] = tensor.from_elements %[[V0]], %[[C1]] : tensor<2xindex>
// CHECK-DAG:     %[[V2:.+]] = shape.value_as_shape %[[V1]] : tensor<2xindex> -> !shape.shape
// CHECK-DAG:     %[[V3:.+]] = shape.value_as_shape %[[C2]] : tensor<1xindex> -> !shape.shape
// CHECK-DAG:     return %[[V2]], %[[V3]], %[[V2]], %[[V2]] : !shape.shape, !shape.shape, !shape.shape, !shape.shape

func @infer_shape_using_dim_op(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>, %arg2: tensor<4x4xf32>) -> !shape.shape {
  %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
  %1 = "mhlo.dot"(%0, %arg2) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[SHAPE0:.*]] = shape.shape_of %arg0 : tensor<?x4xf32> -> tensor<2xindex>
  // CHECK-DAG: %[[DIM0:.*]] = tensor.extract %[[SHAPE0]][%[[C0]]] : tensor<2xindex>
  // CHECK-DAG: %[[SHAPE:.*]] = tensor.from_elements %[[DIM0]], %[[C4]] : tensor<2xindex>
  %2 = shape.shape_of %1 : tensor<?x4xf32> -> tensor<2xindex>
  // CHECK-DAG: %[[V0:.*]] = shape.value_as_shape %[[SHAPE]] : tensor<2xindex> -> !shape.shape
  %3 = shape.value_as_shape %2 : tensor<2xindex> -> !shape.shape
  return %3 : !shape.shape
}

// TODO: Check this after nested function call is supported
func private @inner_func(%arg0 : tensor<?x4xf32>, %arg1 : tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}
func @outer_func(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>) -> (!shape.shape, !shape.shape) {
  %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
  %1 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
  %2 = shape.value_as_shape %1 : tensor<2xindex> -> !shape.shape
  %3 = call @inner_func(%0, %arg0) : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %4 = shape.shape_of %3 : tensor<?x4xf32> -> tensor<2xindex>
  %5 = shape.value_as_shape %4 : tensor<2xindex> -> !shape.shape
  return %2, %5 : !shape.shape, !shape.shape
}
