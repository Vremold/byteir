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

func.func private @aten.max.456(%arg0: tensor<1x1x3xf32>) -> tensor<1x1xf32> {
  %1 = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
  %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [2] : (tensor<1x1x3xf32>, tensor<f32>) -> tensor<1x1xf32>
    reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
    %9 = mhlo.maximum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%9) : (tensor<f32>) -> ()
  }
  return %2 : tensor<1x1xf32>
}

func.func @torch_max_one_result(%arg0: tensor<1x1x3xf32>) -> tensor<1x1xf32> {
  %0 = call @aten.max.456(%arg0) : (tensor<1x1x3xf32>) -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @torch_max_one_result
// CHECK-NEXT: mhlo.custom_call
// CHECK-SAME: call_target_name = "byteir.reduce_max"
// CHECK-NOT: call @aten.max.456

func.func private @aten.max.321(%arg0: tensor<1x1x3xf32>) -> (tensor<1x1xf32>, tensor<1x1xi64>) {
  %0 = mhlo.constant dense<3> : tensor<i64>
  %1 = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
  %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [2] : (tensor<1x1x3xf32>, tensor<f32>) -> tensor<1x1xf32>
    reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
    %9 = mhlo.maximum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%9) : (tensor<f32>) -> ()
  }
  %3 = "mhlo.iota"() {iota_dimension = 2 : i64} : () -> tensor<1x1x3xi32>
  %4 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %5 = mhlo.constant dense<0> : tensor<i32>
  %6:2 = mhlo.reduce(%arg0 init: %4), (%3 init: %5) across dimensions = [2] : (tensor<1x1x3xf32>, tensor<1x1x3xi32>, tensor<f32>, tensor<i32>) -> (tensor<1x1xf32>, tensor<1x1xi32>)
    reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
    %9 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %10 = "mhlo.select"(%9, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = mhlo.minimum %arg2, %arg4 : tensor<i32>
    %13 = "mhlo.select"(%9, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %14 = "mhlo.select"(%11, %12, %13) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%10, %14) : (tensor<f32>, tensor<i32>) -> ()
  }
  %7 = mhlo.convert(%6#1) : (tensor<1x1xi32>) -> tensor<1x1xi64>
  return %2, %7 : tensor<1x1xf32>, tensor<1x1xi64>
}

func.func @torch_max_two_results(%arg0: tensor<1x1x3xf32>) -> (tensor<1x1xf32>, tensor<1x1xi64>) {
  %0:2 = call @aten.max.321(%arg0) : (tensor<1x1x3xf32>) -> (tensor<1x1xf32>, tensor<1x1xi64>)
  return %0#0, %0#1: tensor<1x1xf32>, tensor<1x1xi64>
}
// CHECK-LABEL: func.func @torch_max_two_results
// CHECK-NEXT: mhlo.custom_call
// CHECK-SAME: call_target_name = "byteir.arg_max"
// CHECK-NOT: call @aten.max.321

func.func private @aten.max.2323(%arg0: tensor<1x1x3xf32>) -> tuple<tensor<1x1xf32>, tensor<1x1xi64>> {
  %0 = mhlo.constant dense<3> : tensor<i64>
  %1 = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
  %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [2] : (tensor<1x1x3xf32>, tensor<f32>) -> tensor<1x1xf32>
    reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
    %9 = mhlo.maximum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%9) : (tensor<f32>) -> ()
  }
  %3 = "mhlo.iota"() {iota_dimension = 2 : i64} : () -> tensor<1x1x3xi32>
  %4 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %5 = mhlo.constant dense<0> : tensor<i32>
  %6:2 = mhlo.reduce(%arg0 init: %4), (%3 init: %5) across dimensions = [2] : (tensor<1x1x3xf32>, tensor<1x1x3xi32>, tensor<f32>, tensor<i32>) -> (tensor<1x1xf32>, tensor<1x1xi32>)
    reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
    %9 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %10 = "mhlo.select"(%9, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = mhlo.minimum %arg2, %arg4 : tensor<i32>
    %13 = "mhlo.select"(%9, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %14 = "mhlo.select"(%11, %12, %13) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%10, %14) : (tensor<f32>, tensor<i32>) -> ()
  }
  %7 = mhlo.convert(%6#1) : (tensor<1x1xi32>) -> tensor<1x1xi64>
  %8 = "mhlo.tuple"(%2, %7) {xla_shape = "(f32[1,1]{1,0}, s64[1,1]{1,0})"} : (tensor<1x1xf32>, tensor<1x1xi64>) -> tuple<tensor<1x1xf32>, tensor<1x1xi64>>
  return %8 : tuple<tensor<1x1xf32>, tensor<1x1xi64>>
}

func.func @torch_max_tuple(%arg0: tensor<1x1x3xf32>) -> tuple<tensor<1x1xf32>, tensor<1x1xi64>> {
  %0 = call @aten.max.2323(%arg0) : (tensor<1x1x3xf32>) -> tuple<tensor<1x1xf32>, tensor<1x1xi64>>
  return %0 : tuple<tensor<1x1xf32>, tensor<1x1xi64>>
}
// CHECK-LABEL: func.func @torch_max_tuple
// CHECK-NEXT: mhlo.custom_call
// CHECK-SAME: call_target_name = "byteir.arg_max"
// CHECK-NOT: call @aten.max.321