// RUN: byteir-opt -hlo-opt %s | FileCheck %s

func @uniform_rng() -> tensor<2x128x128xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<[2, 128, 128]> : tensor<3xi64>
    %3 = "mhlo.rng_uniform"(%0, %1, %2) : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x128x128xf32>
    %4 = "mhlo.rng_uniform"(%0, %1, %2) : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x128x128xf32>
    %5 = mhlo.add %3, %4 : tensor<2x128x128xf32>
    return %5 : tensor<2x128x128xf32>
}

// CHECK-LABEL: func private @RngUniform0
//   CHECK-DAG: __byre__high = 1.000000e+00
//   CHECK-DAG: __byre__low = 0.000000e+00
//   CHECK-DAG: byre_compute_name = "RngUniform"
//   CHECK-DAG: byre_force_compute_name

// CHECK-LABEL: func private @RngUniform1
//   CHECK-DAG: __byre__high = 1.000000e+00
//   CHECK-DAG: __byre__low = 0.000000e+00
//   CHECK-DAG: byre_compute_name = "RngUniform"
//   CHECK-DAG: byre_force_compute_name

// CHECK-LABEL: func @uniform_rng
//   CHECK: %[[VAR_0:.*]] = call @RngUniform0
//   CHECK: %[[VAR_1:.*]] = call @RngUniform1
//   CHECK: %[[VAR_2:.*]] = mhlo.add %[[VAR_0]], %[[VAR_1]]
//   CHECK: return %[[VAR_2]]
