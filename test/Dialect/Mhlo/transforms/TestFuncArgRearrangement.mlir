// RUN: byteir-opt %s -test-rearrange-func-arg -canonicalize-ext | FileCheck %s

func.func private @test_func(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x5xf32>, %arg4: tensor<4x2x3xf32>, %arg5: tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>, tensor<128x5xf32>, tensor<128x2x3xf32>) attributes {__byteir_unit_test__ = {arg = [["pack", 3, 0, 2], ["identity", 1], ["pack2d", 5, 4]], result = [["pack", 2, 0], ["identity", 1], ["pack2d", 4, 3]]}, other_attr_1}
// CHECK-LABEL: func.func private @test_func
// CHECK-SAME: (tensor<4x11xf32>, tensor<4x3xf32>, tensor<4x13xf32>) -> (tensor<128x6xf32>, tensor<128x3xf32>, tensor<128x11xf32>) attributes {other_attr_1}


func.func @main1(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x5xf32>, %arg4: tensor<4x2x3xf32>, %arg5: tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>, tensor<128x5xf32>, tensor<128x2x3xf32>) attributes {__byteir_unit_test__ = {arg = [["pack", 3, 0, 2], ["identity", 1], ["pack2d", 5, 4]], result = [["pack", 2, 0], ["identity", 1], ["pack2d", 4, 3]]}, other_attr_2} {
  %0:5 = call @test_func(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (tensor<4x2xf32>, tensor<4x3xf32>, tensor<4x4xf32>, tensor<4x5xf32>, tensor<4x2x3xf32>, tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>, tensor<128x5xf32>, tensor<128x2x3xf32>)
  return %0#0, %0#1, %0#2, %0#3, %0#4 : tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>, tensor<128x5xf32>, tensor<128x2x3xf32>
}
// CHECK-LABEL: func.func @main1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x11xf32>, %[[ARG1:.*]]: tensor<4x3xf32>, %[[ARG2:.*]]: tensor<4x13xf32>) -> (tensor<128x6xf32>, tensor<128x3xf32>, tensor<128x11xf32>) attributes {other_attr_2}
// CHECK-NEXT: %[[V0:.*]]:3 = call @test_func(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (tensor<4x11xf32>, tensor<4x3xf32>, tensor<4x13xf32>) -> (tensor<128x6xf32>, tensor<128x3xf32>, tensor<128x11xf32>)
// CHECK-NEXT: return %[[V0:.*]]#0, %[[V0:.*]]#1, %[[V0:.*]]#2 : tensor<128x6xf32>, tensor<128x3xf32>, tensor<128x11xf32>


