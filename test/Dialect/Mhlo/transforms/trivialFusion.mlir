// RUN: byteir-opt %s -fuse-trivial | FileCheck %s

func @test_rng_uniform(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<3xi64>) -> tensor<2x128x128xf32> {
  %0 = "mhlo.rng_uniform"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x128x128xf32>
  return %0 : tensor<2x128x128xf32>
}
// CHECK-LABEL: func @test_rng_uniform
// CHECK:  mhlo.fusion
// CHECK-NEXT:  mhlo.rng_uniform
// CHECK-NEXT:  mhlo.return
// CHECK-NEXT:  __byteir_trivial_fusion__, byre_compute_name = "RngUniform"

