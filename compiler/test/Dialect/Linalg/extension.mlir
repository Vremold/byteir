// RUN: byteir-opt %s | FileCheck %s

func.func @custom_memref(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) {
  linalg_ext.custom  {target_name = "foo"} ins(%arg0 : memref<1024x64xf32>) outs(%arg1 : memref<1024x64xf32>) {
    ^bb0(%arg2 : memref<1024x64xf32>, %arg3 : memref<1024x64xf32>):  // no predecessors
      "lmhlo.custom_call"(%arg2, %arg3) {call_target_name = "bar", has_side_effect = false, operand_segment_sizes = array<i32: 1, 1>} : (memref<1024x64xf32>, memref<1024x64xf32>) -> ()
      linalg_ext.yield
  }
  return
}
//CHECK-LABEL: func.func @custom_memref
//CHECK: linalg_ext.custom
//CHECK: lmhlo.custom_call

func.func @custom_tensor(%arg0: tensor<1024x64xf32>) -> (tensor<1024x64xf32>) {
  %0 = linalg_ext.custom {target_name = "foo"} outs(%arg0 : tensor<1024x64xf32>) {
    ^bb0(%arg1 : tensor<1024x64xf32>):  // no predecessors
      %1 = "mhlo.custom_call"(%arg1) {call_target_name = "bar", has_side_effect = false} : (tensor<1024x64xf32>) -> tensor<1024x64xf32>
      linalg_ext.yield %1 : tensor<1024x64xf32>
  } -> tensor<1024x64xf32>
  return %0 : tensor<1024x64xf32>
}
//CHECK-LABEL: func.func @custom_tensor
//CHECK: linalg_ext.custom
//CHECK: mhlo.custom_call

func.func @scan_1d_tensor(%0: tensor<128xi32>) -> tensor<128xi32> {
  %c0 = tensor.empty() : tensor<i32>
  %1 = tensor.empty() : tensor<128xi32>
  %2:2 = linalg_ext.scan
    dimension(0) inclusive(true)
    ins(%0 : tensor<128xi32>) outs(%1, %c0 : tensor<128xi32>, tensor<i32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      linalg_ext.yield %sum : i32
  } -> tensor<128xi32>, tensor<i32>
  return %2#0 : tensor<128xi32>
}
//CHECK-LABEL: func.func @scan_1d_tensor
//CHECK: linalg_ext.scan

func.func @scan_2d_tensor(%0: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %c0 = tensor.empty() : tensor<32xi32>
  %1 = tensor.empty() : tensor<16x32xi32>
  %2:2 = linalg_ext.scan
    dimension(0) inclusive(true)
    ins(%0 : tensor<16x32xi32>) outs(%1, %c0 : tensor<16x32xi32>, tensor<32xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      linalg_ext.yield %sum : i32
  } -> tensor<16x32xi32>, tensor<32xi32>
  return %2#0 : tensor<16x32xi32>
}
//CHECK-LABEL: func.func @scan_2d_tensor
//CHECK: linalg_ext.scan

func.func @scan_2d_memref(%0: memref<16x32xi32>, %1: memref<16x32xi32>) {
  %c0 = memref.alloc() : memref<32xi32>
  linalg_ext.scan
    dimension(0) inclusive(true)
    ins(%0 : memref<16x32xi32>) outs(%1, %c0 : memref<16x32xi32>, memref<32xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      linalg_ext.yield %sum : i32
  }
  return
}
//CHECK-LABEL: func.func @scan_2d_memref
//CHECK: linalg_ext.scan

func.func @softmax_tensor(%arg0: tensor<1024x64xf32>) -> (tensor<1024x64xf32>) {
  %0 = tensor.empty() : tensor<1024x64xf32>
  %1 = tensor.empty() : tensor<64xf32>
  %2 = tensor.empty() : tensor<64xf32>
  %3 = tensor.empty() : tensor<64xf32>
  %4:4 = linalg_ext.softmax 
    dimension(0) 
    ins(%arg0 : tensor<1024x64xf32>) outs(%0, %1, %2, %3 : tensor<1024x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) : tensor<1024x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
  return %4#0 : tensor<1024x64xf32>
}
//CHECK-LABEL: func.func @softmax_tensor
//CHECK: linalg_ext.softmax

func.func @softmax_memref(%arg0: memref<1024x64xf32>) -> (memref<1024x64xf32>) {
  %0 = memref.alloc() : memref<1024x64xf32>
  %1 = memref.alloc() : memref<64xf32>
  %2 = memref.alloc() : memref<64xf32>
  %3 = memref.alloc() : memref<64xf32>
  linalg_ext.softmax 
    dimension(0)
    ins(%arg0 : memref<1024x64xf32>) outs(%0, %1, %2, %3 : memref<1024x64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>)
  return %0 : memref<1024x64xf32>
}
//CHECK-LABEL: func.func @softmax_memref
//CHECK: linalg_ext.softmax