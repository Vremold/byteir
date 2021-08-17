// RUN: byteir-opt -allow-unregistered-dialect %s | FileCheck %s

module attributes {byre.container_module} {
  func @test_compute(%arg0: memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    byre.compute @some_kernel(%arg0, %arg1) : memref<100x?xf32>, memref<100x?xf32>
    return
  }
// CHECK-LABEL: func @test_compute
// CHECK: %arg0: memref<100x?xf32>
//  CHECK-DAG: byre.argtype = 1 : i32
//  CHECK-DAG: byre.argname = "A"
// CHECK: %arg1: memref<100x?xf32>
//  CHECK-DAG: byre.argtype = 2 : i32
//  CHECK-DAG: byre.argname = "B"
// CHECK: attributes {byre.entry_point} {
// CHECK:   byre.compute @some_kernel(%arg0, %arg1) : memref<100x?xf32>, memref<100x?xf32>
// CHECK:   return
// CHECK: }


  func @test_copy(%arg0 : memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1 : memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    byre.copy(%arg0, %arg1) : memref<100x?xf32>, memref<100x?xf32>
    return
  }
// CHECK-LABEL: func @test_copy
// CHECK: %arg0: memref<100x?xf32>
//  CHECK-DAG: byre.argtype = 1 : i32
//  CHECK-DAG: byre.argname = "A"
// CHECK: %arg1: memref<100x?xf32> {
//  CHECK-DAG: byre.argtype = 2 : i32
//  CHECK-DAG: byre.argname = "B"
// CHECK: attributes {byre.entry_point} {
// CHECK:   byre.copy(%arg0, %arg1) : memref<100x?xf32>, memref<100x?xf32>
// CHECK:   return
// CHECK: }

}

module attributes {byre.container_module, byre.memory_space = [1, "CPU", 12, "CUDA"]} {
// CHECK: module attributes {byre.container_module, byre.memory_space = [1, "CPU", 12, "CUDA"]}
  func @dummy() {
    return 
  }
}
