// RUN: byteir-opt --convert-hlo-to-lhlo %s | FileCheck %s

func @clamp(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "mhlo.clamp"(%arg0, %arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}
// CHECK: lmhlo.clamp
