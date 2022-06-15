// RUN: byteir-opt %s -set-op-space="entry-func=main space=cpu" -set-arg-space="entry-func=main all-space=cpu" | FileCheck %s

module {
  func @main(%arg0: memref<128x64xf32>, %arg1: memref<64x32xf32>, %arg2: memref<32xf32>) -> memref<128x32xf32> {
    %0 = memref.alloc() : memref<128x64xf16>
    "lmhlo.convert"(%arg0, %0) : (memref<128x64xf32>, memref<128x64xf16>) -> ()
    %1 = memref.alloc() : memref<64x32xf16>
    "lmhlo.convert"(%arg1, %1) : (memref<64x32xf32>, memref<64x32xf16>) -> ()
    %2 = memref.alloc() : memref<32xf16>
    "lmhlo.convert"(%arg2, %2) : (memref<32xf32>, memref<32xf16>) -> ()
    %3 = call @mlp_device(%0, %1, %2) : (memref<128x64xf16>, memref<64x32xf16>, memref<32xf16>) -> memref<128x32xf16>
    %4 = memref.alloc() : memref<128x32xf32>
    "lmhlo.convert"(%3, %4) : (memref<128x32xf16>, memref<128x32xf32>) -> ()
    return %4 : memref<128x32xf32>
  }
  func private @mlp_device(memref<128x64xf16>, memref<64x32xf16>, memref<32xf16>) -> memref<128x32xf16> attributes {device = "test"}
}
// CHECK-LABEL: func @main(%arg0: memref<128x64xf32, "cpu">, %arg1: memref<64x32xf32, "cpu">, %arg2: memref<32xf32, "cpu">) -> memref<128x32xf32, "cpu"> {
// CHECK-NEXT:    %0 = memref.alloc() : memref<128x64xf16, "test">
// CHECK-NEXT:    %1 = memref.alloc() : memref<128x64xf16, "cpu">
// CHECK-NEXT:    "lmhlo.convert"(%arg0, %1) {device = "cpu"} : (memref<128x64xf32, "cpu">, memref<128x64xf16, "cpu">) -> ()
// CHECK-NEXT:    memref.copy %1, %0 : memref<128x64xf16, "cpu"> to memref<128x64xf16, "test">
// CHECK-NEXT:    %2 = memref.alloc() : memref<64x32xf16, "test">
// CHECK-NEXT:    %3 = memref.alloc() : memref<64x32xf16, "cpu">
// CHECK-NEXT:    "lmhlo.convert"(%arg1, %3) {device = "cpu"} : (memref<64x32xf32, "cpu">, memref<64x32xf16, "cpu">) -> ()
// CHECK-NEXT:    memref.copy %3, %2 : memref<64x32xf16, "cpu"> to memref<64x32xf16, "test">
// CHECK-NEXT:    %4 = memref.alloc() : memref<32xf16, "test">
// CHECK-NEXT:    %5 = memref.alloc() : memref<32xf16, "cpu">
// CHECK-NEXT:    lmhlo.convert"(%arg2, %5) {device = "cpu"} : (memref<32xf32, "cpu">, memref<32xf16, "cpu">) -> ()
// CHECK-NEXT:    memref.copy %5, %4 : memref<32xf16, "cpu"> to memref<32xf16, "test">
// CHECK-NEXT:    %6 = call @mlp_device(%0, %2, %4) : (memref<128x64xf16, "test">, memref<64x32xf16, "test">, memref<32xf16, "test">) -> memref<128x32xf16, "test">
// CHECK-NEXT:    %7 = memref.alloc() : memref<128x32xf32, "cpu">
// CHECK-NEXT:    %8 = memref.alloc() : memref<128x32xf16, "cpu">
// CHECK-NEXT:    memref.copy %6, %8 : memref<128x32xf16, "test"> to memref<128x32xf16, "cpu">
// CHECK-NEXT:    "lmhlo.convert"(%8, %7) {device = "cpu"} : (memref<128x32xf16, "cpu">, memref<128x32xf32, "cpu">) -> ()
// CHECK-NEXT:    return %7 : memref<128x32xf32, "cpu">
