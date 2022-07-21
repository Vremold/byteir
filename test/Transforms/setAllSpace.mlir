// RUN: byteir-opt %s -set-op-space="entry-func=main space=cpu" -set-arg-space="entry-func=main all-space=cpu" | FileCheck %s

func @main(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32>, %arg2 : memref<2x4xf32>) -> (memref<2x4xf32>, memref<2x4xf32>) {
  %0 = memref.alloc() : memref<2x4xf32>
  "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>) -> ()
  "lmhlo.add"(%arg0, %arg1, %0) : (memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>) -> ()
  return %0, %0: memref<2x4xf32>, memref<2x4xf32> 
}
// CHECK-LABEL: func @main(%arg0: memref<2x4xf32, "cpu">, %arg1: memref<2x4xf32, "cpu">, %arg2: memref<2x4xf32, "cpu">) -> (memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">) 
// CHECK-NEXT:     %0 = memref.alloc() : memref<2x4xf32, "cpu">
// CHECK-NEXT:     "lmhlo.add"(%arg0, %arg1, %arg2) {device = "cpu"} : (memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">) -> ()
// CHECK-NEXT:     "lmhlo.add"(%arg0, %arg1, %0) {device = "cpu"} : (memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">) -> ()
// CHECK-NEXT:     return %0, %0 : memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">
