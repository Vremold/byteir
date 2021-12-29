// RUN: byteir-opt -byre-fold %s | FileCheck %s

module attributes {byre.container_module} {
  func @fold_alias(%arg0: memref<512xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<512xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<128xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<512xf32>
    %5 = memref.alloc() : memref<256xf32>
    %6 = memref.alloc() : memref<128xf32>
    byre.compute @AliasOp(%arg0, %0) {arg_alias, offset = 128 : i32} : memref<512xf32>, memref<256xf32>
    byre.compute @AliasOp(%0, %1) {offset = 32 : i32} : memref<256xf32>, memref<128xf32>
// CHECK: byre.compute @AliasOp(%arg0, %1) {arg_alias, offset = 160 : i32}
    byre.compute @AliasOp(%arg1, %2) {arg_alias, offset = 128 : i32} : memref<512xf32>, memref<256xf32>
    byre.compute @AliasOp(%2, %3) {offset = 32 : i32} : memref<256xf32>, memref<128xf32>
// CHECK: byre.compute @AliasOp(%arg1, %3) {arg_alias, offset = 160 : i32}
    byre.compute @AliasOp(%4, %5) {offset = 128 : i32} : memref<512xf32>, memref<256xf32>
    byre.compute @AliasOp(%5, %6) {offset = 32 : i32} : memref<256xf32>, memref<128xf32>
// CHECK: byre.compute @AliasOp(%4, %6) {offset = 160 : i32}
    return
  }
}
