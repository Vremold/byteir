// RUN: byteir-opt %s -set-op-space="entry-func=main space=cpu" -set-arg-space="entry-func=main all-space=cpu" | FileCheck %s

module {
  func.func @main(%arg0: memref<128x64xf32>, %arg1: memref<64x32xf32>, %arg2: memref<32xf32>) -> memref<128x32xf32> {
    %0 = memref.alloc() : memref<128x64xf16>
    "lmhlo.convert"(%arg0, %0) : (memref<128x64xf32>, memref<128x64xf16>) -> ()
    %1 = memref.alloc() : memref<64x32xf16>
    "lmhlo.convert"(%arg1, %1) : (memref<64x32xf32>, memref<64x32xf16>) -> ()
    %2 = memref.alloc() : memref<32xf16>
    "lmhlo.convert"(%arg2, %2) : (memref<32xf32>, memref<32xf16>) -> ()
    %3 = call @func_with_bufferization() : () -> memref<1x97xf32>
    %4 = call @mlp_device(%0, %1, %2, %3) : (memref<128x64xf16>, memref<64x32xf16>, memref<32xf16>, memref<1x97xf32>) -> memref<128x32xf16>
    %5 = memref.alloc() : memref<128x32xf32>
    "lmhlo.convert"(%4, %5) : (memref<128x32xf16>, memref<128x32xf32>) -> ()
    return %5 : memref<128x32xf32>
  }
  func.func private @mlp_device(memref<128x64xf16>, memref<64x32xf16>, memref<32xf16>, memref<1x97xf32>) -> memref<128x32xf16> attributes {device = "test"}

  func.func private @func_with_bufferization() -> memref<1x97xf32> attributes {device = "cpu"} {
    %0 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%0) {device = "cpu", value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %1 = bufferization.to_tensor %0 {device = "cpu"} : memref<f32>
    %2 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%2) {device = "cpu", value = dense<1.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %3 = bufferization.to_tensor %2 {device = "cpu"} : memref<f32>
    %4 = memref.alloc() : memref<2xi64>
    "lmhlo.constant"(%4) {device = "cpu", value = dense<[1, 97]> : tensor<2xi64>} : (memref<2xi64>) -> ()
    %5 = bufferization.to_tensor %4 {device = "cpu"} : memref<2xi64>
    %6 = "mhlo.rng"(%1, %3, %5) {device = "host", rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<1x97xf32>
    %7 = bufferization.to_memref %6 {device = "cpu"} : memref<1x97xf32>
    return %7 : memref<1x97xf32>
  }
}
// CHECK-LABEL: func.func @main(%arg0: memref<128x64xf32, "cpu">, %arg1: memref<64x32xf32, "cpu">, %arg2: memref<32xf32, "cpu">) -> memref<128x32xf32, "cpu"> {
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
// CHECK-NEXT:    "lmhlo.convert"(%arg2, %5) {device = "cpu"} : (memref<32xf32, "cpu">, memref<32xf16, "cpu">) -> ()
// CHECK-NEXT:    memref.copy %5, %4 : memref<32xf16, "cpu"> to memref<32xf16, "test">
// CHECK-NEXT:    %6 = call @func_with_bufferization() : () -> memref<1x97xf32, "cpu">
// CHECK-NEXT:    %7 = memref.alloc() : memref<1x97xf32, "test">
// CHECK-NEXT:    memref.copy %6, %7 : memref<1x97xf32, "cpu"> to memref<1x97xf32, "test">
// CHECK-NEXT:    %8 = call @mlp_device(%0, %2, %4, %7) : (memref<128x64xf16, "test">, memref<64x32xf16, "test">, memref<32xf16, "test">, memref<1x97xf32, "test">) -> memref<128x32xf16, "test">
// CHECK-NEXT:    %9 = memref.alloc() : memref<128x32xf32, "cpu">
// CHECK-NEXT:    %10 = memref.alloc() : memref<128x32xf16, "cpu">
// CHECK-NEXT:    memref.copy %8, %10 : memref<128x32xf16, "test"> to memref<128x32xf16, "cpu">
// CHECK-NEXT:    "lmhlo.convert"(%10, %9) {device = "cpu"} : (memref<128x32xf16, "cpu">, memref<128x32xf32, "cpu">) -> ()
// CHECK-NEXT:    return %9 : memref<128x32xf32, "cpu">
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @mlp_device(memref<128x64xf16, "test">, memref<64x32xf16, "test">, memref<32xf16, "test">, memref<1x97xf32, "test">) -> memref<128x32xf16, "test"> attributes {device = "test"}

// CHECK-LABEL: func.func private @func_with_bufferization() -> memref<1x97xf32, "cpu"> attributes {device = "cpu"} {
// CHECK-NEXT:    %0 = memref.alloc() : memref<f32, "cpu">
// CHECK-NEXT:    "lmhlo.constant"(%0) {device = "cpu", value = dense<0.000000e+00> : tensor<f32>} : (memref<f32, "cpu">) -> ()
// CHECK-NEXT:    %1 = bufferization.to_tensor %0 {device = "cpu"} : memref<f32, "cpu">
// CHECK-NEXT:    %2 = memref.alloc() : memref<f32, "cpu">
// CHECK-NEXT:    "lmhlo.constant"(%2) {device = "cpu", value = dense<1.000000e+00> : tensor<f32>} : (memref<f32, "cpu">) -> ()
// CHECK-NEXT:    %3 = bufferization.to_tensor %2 {device = "cpu"} : memref<f32, "cpu">
// CHECK-NEXT:    %4 = memref.alloc() : memref<2xi64, "cpu">
// CHECK-NEXT:    "lmhlo.constant"(%4) {device = "cpu", value = dense<[1, 97]> : tensor<2xi64>} : (memref<2xi64, "cpu">) -> ()
// CHECK-NEXT:    %5 = bufferization.to_tensor %4 {device = "cpu"} : memref<2xi64, "cpu">
// CHECK-NEXT:    %6 = "mhlo.rng"(%1, %3, %5) {device = "host", rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<1x97xf32>
// CHECK-NEXT:    %7 = bufferization.to_memref %6 {device = "cpu"} : memref<1x97xf32, "cpu">
// CHECK-NEXT:    return %7 : memref<1x97xf32, "cpu">
// CHECK-NEXT:  }
