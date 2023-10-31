// RUN: byteir-opt --byre-to-byre-serial %s | byteir-opt --byre-serial-to-byre | FileCheck %s
// RUN: byteir-opt --byre-to-byre-serial -emit-bytecode %s | byteir-opt --byre-serial-to-byre | FileCheck %s
// RUN: byteir-opt --dump-byre="file-name=%t" %s &>/dev/null && byteir-opt -load-byre %t | FileCheck %s
// RUN: byteir-opt --test-byre-serial-round-trip --mlir-disable-threading %s

module attributes {byre.container_module} {
  func.func @test_compute(%arg0: memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B", byre.arg_alias_index = 0 : i64}) attributes {byre.entry_point, byteir.entry_point, tf.original_input_names} {
    byre.compute @some_kernel(%arg0, %arg1) : memref<100x?xf32>, memref<100x?xf32>
    return
  }
// CHECK-LABEL: func.func @test_compute
// CHECK: %arg0: memref<100x?xf32>
//  CHECK-DAG: byre.argtype = 1 : i32
//  CHECK-DAG: byre.argname = "A"
// CHECK: %arg1: memref<100x?xf32>
//  CHECK-DAG: byre.argtype = 2 : i32
//  CHECK-DAG: byre.argname = "B"
//  CHECK-DAG: byre.arg_alias_index = 0 : i64
// CHECK-SAME: attributes {
//  CHECK-DAG:  byre.entry_point
//  CHECK-DAG:  byteir.entry_point
//  CHECK-DAG:  tf.original_input_names
// CHECK-SAME: }
// CHECK:   byre.compute @some_kernel(%arg0, %arg1) : memref<100x?xf32>, memref<100x?xf32>
// CHECK:   return
// CHECK: }


  func.func @test_copy(%arg0 : memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1 : memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    byre.copy(%arg0, %arg1) {callee = "cuda2cuda"} : memref<100x?xf32>, memref<100x?xf32>
    return
  }
// CHECK-LABEL: func.func @test_copy
// CHECK: %arg0: memref<100x?xf32>
//  CHECK-DAG: byre.argtype = 1 : i32
//  CHECK-DAG: byre.argname = "A"
// CHECK: %arg1: memref<100x?xf32> {
//  CHECK-DAG: byre.argtype = 2 : i32
//  CHECK-DAG: byre.argname = "B"
// CHECK: attributes {byre.entry_point} {
// CHECK:   byre.copy(%arg0, %arg1) {callee = "cuda2cuda"} : memref<100x?xf32>, memref<100x?xf32>
// CHECK:   return
// CHECK: }

  func.func @test_group_copy(%arg0 : memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1 : memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}, %arg2 : memref<200x?xf32> {byre.argtype = 1: i32, byre.argname = "C"}, %arg3 : memref<200x?xf32> {byre.argtype = 2: i32, byre.argname = "D"}) attributes {byre.entry_point} {
    "byre.group_copy"(%arg0, %arg2, %arg1, %arg3) {callee = "h2d_array"} : (memref<100x?xf32>, memref<200x?xf32>, memref<100x?xf32>, memref<200x?xf32>) -> ()
    return
  }
// CHECK-LABEL: func.func @test_group_copy
// CHECK: "byre.group_copy"

  func.func @test_alias(%arg0 : memref<100x32xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1 : memref<100x32xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    %0 = "byre.alias"(%arg0) {offset = 0: i64} : (memref<100x32xf32>) -> memref<100x32xf32>
    byre.compute @some_kernel(%0, %arg1) : memref<100x32xf32>, memref<100x32xf32>
    return
  }
// CHECK-LABEL: func.func @test_alias
// CHECK: "byre.alias"

  func.func @test_tensor_attr(%arg0: memref<100x100xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    byre.compute @some_kernel(%arg0, %arg1) {weight = dense<1.0> : tensor<100x100xf32>, value = dense<"-1"> : tensor<100x!ace.string>} : memref<100x100xf32>, memref<100x?xf32>
    return
  }
// CHECK-LABEL: func.func @test_tensor_attr
// CHECK-NEXT: byre.compute @some_kernel
// CHECK-SAME: value = dense<"-1"> : tensor<100x!ace.string>
// CHECK-SAME: weight = dense<1.000000e+00> : tensor<100x100xf32>

  func.func @test_dense_array_attr(%arg0: memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    byre.compute @some_kernel(%arg0, %arg1) {memory_effects = [1 : i32, 1 : i32], dense_array_attr = array<i32: 10, 42>, integer_attr = 1 : i32, float_attr = 1.0 : f32} : memref<100x?xf32>, memref<100x?xf32>
    return
  }
// CHECK-LABEL: func.func @test_dense_array_attr
// CHECK: dense_array_attr = array<i32: 10, 42>
// CHECK-SAME: float_attr = 1.000000e+00 : f32
// CHECK-SAME: integer_attr = 1 : i32
}
