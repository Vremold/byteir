// RUN: byteir-opt %s | FileCheck %s

func @mhlo_add(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
    %res = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %res : tensor<4xf32>
}
// CHECK-LABEL: func @mhlo_add
