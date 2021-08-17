// RUN: byteir-opt --convert-to-byre %s | FileCheck %s

module {
// CHECK: module attributes {byre.container_module}  {
  func private @some_func(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32> attributes { byre_compute_name = "customAddOp"}  {
    %0 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %arg1, %0) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    %1 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %0, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return %1 : memref<4xf32>
  }

  func @mhlo_add(%arg0: memref<4xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<4xf32> {__placeholder__byre.argname = "B"}) -> (memref<4xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point } {
    %0 = call @some_func(%arg0, %arg1) : (memref<4xf32>, memref<4xf32>) -> memref<4xf32>
    return %0 : memref<4xf32>
  }
// CHECK:   func @mhlo_add(%[[ARG_0:.*]]: memref<4xf32> {byre.argname = "A", byre.argtype = 1 : i32}, %[[ARG_1:.*]]: memref<4xf32> {byre.argname = "B", byre.argtype = 1 : i32}, %[[ARG_2:.*]]: memref<4xf32> {byre.argname = "C", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK:     byre.compute @customAddOp(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]]) : memref<4xf32>, memref<4xf32>, memref<4xf32>
// CHECK:     return
}




