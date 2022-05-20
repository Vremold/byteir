// RUN: byteir-opt %s -set-arg-space="entry-func=main all-space=cpu" | FileCheck %s


func private @nested(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32>)
// CHECK-LABEL: func private @nested(memref<2x4xf32, "gpu">, memref<2x4xf32, "gpu">)

func private @local(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32>) attributes {device = "gpu"} {
  call @nested(%arg0, %arg1) : (memref<2x4xf32>, memref<2x4xf32>) -> ()
  "lmhlo.abs"(%arg0, %arg1) : (memref<2x4xf32>, memref<2x4xf32>) -> ()
  return 
}
// CHECK-LABEL: func private @local(%arg0: memref<2x4xf32, "gpu">, %arg1: memref<2x4xf32, "gpu">)
// CHECK-NEXT:    call @nested(%arg0, %arg1) : (memref<2x4xf32, "gpu">, memref<2x4xf32, "gpu">) -> ()
// CHECK-NEXT:    "lmhlo.abs"(%arg0, %arg1) : (memref<2x4xf32, "gpu">, memref<2x4xf32, "gpu">) -> ()

func @main(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32>, %arg2 : memref<2x4xf32>) {
  %0 = memref.alloc() : memref<2x4xf32>
  call @local(%arg0, %0) : (memref<2x4xf32>, memref<2x4xf32>) -> ()
  "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>) -> ()
  return
}
// CHECK-LABEL: func @main(%arg0: memref<2x4xf32, "cpu">, %arg1: memref<2x4xf32, "cpu">, %arg2: memref<2x4xf32, "cpu">)
// CHECK-NEXT:    %0 = memref.alloc() : memref<2x4xf32, "gpu">
// CHECK-NEXT:    %1 = memref.alloc() : memref<2x4xf32, "gpu">
// CHECK-NEXT:    memref.copy %arg0, %1 : memref<2x4xf32, "cpu"> to memref<2x4xf32, "gpu">
// CHECK-NEXT:    call @local(%1, %0) : (memref<2x4xf32, "gpu">, memref<2x4xf32, "gpu">) -> ()
// CHECK-NEXT:    "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">) -> ()

