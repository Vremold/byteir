// RUN: byteir-opt %s -convert-func-to-custom-torch-xla | FileCheck %s

func.func private @aten.erf.123(%arg0: tensor<4x?xf32>) -> tensor<4x?xf32>
// CHECK-NOT: func.func private @aten.gelu.123
func.func private @aten.other.456(%arg0: tensor<4x?xf32>) -> tensor<4x?xf32>
// CHECK-LABEL: func.func private @aten.other.456

func.func @main1(%arg0: tensor<4x?xf32>) -> tensor<4x?xf32> {
  %0 = call @aten.erf.123(%arg0) : (tensor<4x?xf32>) -> tensor<4x?xf32>
  return %0 : tensor<4x?xf32>
}
// CHECK-LABEL:  func.func @main1
// CHECK-NEXT: mhlo.custom_call
// CHEKC-SAME: call_target_name = "byteir.erf"
// CHECK-NOT: call @aten.erf.123


func.func @main2(%arg0: tensor<4x?xf32>) -> tensor<4x?xf32> {
  %0 = call @aten.other.456(%arg0) : (tensor<4x?xf32>) -> tensor<4x?xf32>
  return %0 : tensor<4x?xf32>
}
// CHECK-LABEL:  func.func @main2
// CHECK-NEXT: call @aten.other.456
