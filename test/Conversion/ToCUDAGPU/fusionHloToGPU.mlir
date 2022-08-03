// RUN: byteir-opt -hlo-legalize-to-linalg -linalg-fuse-elementwise-ops -linalg-bufferize -func-bufferize -canonicalize -cse -sccp -pass-pipeline="func.func(linalg-detensorize)" -convert-linalg-to-affine-loops -rewrite-affine-to-memref -pass-pipeline="func.func(convert-affine-for-to-gpu)" -cse -lower-affine -gpu-kernel-outlining -cse %s | FileCheck %s

func.func @fusion_broadcast(%arg0: tensor<6x12x96xf32>, %arg1: tensor<6x12x96x96xf32>) -> tensor<6x12x96x96xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<6x12x96xf32>) -> tensor<6x12x96x96xf32>
  %1 = mhlo.subtract %arg1, %0 : tensor<6x12x96x96xf32>
  %2 = "mhlo.exponential"(%1) : (tensor<6x12x96x96xf32>) -> tensor<6x12x96x96xf32>
  return %2 : tensor<6x12x96x96xf32>
}

// CHECK-LABEL: fusion_broadcast
// CHECK: gpu.launch_func
// CHECK: gpu.func @fusion_broadcast_kernel