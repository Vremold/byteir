// RUN: byteir-opt %s -linalg-opt | FileCheck %s

// CHECK-LABEL: func @main
module {
  func private @MatmulOp0(%arg0: tensor<256x128xf32>, %arg1: tensor<256x30522xf32>) -> tensor<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256x128xf32>, tensor<256x30522xf32>) -> tensor<128x30522xf32>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %1 : tensor<30522x128xf32>
  }
  func private @Unknown0(%arg0: tensor<2x128xi64>) -> (tensor<256xi64>, tensor<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<-100> : tensor<256xi64>
    %1 = "mhlo.reshape"(%arg0) : (tensor<2x128xi64>) -> tensor<256xi64>
    %2 = "mhlo.compare"(%1, %0) {comparison_direction = "NE"} : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    return %1, %2 : tensor<256xi64>, tensor<256xi1>
  }
  func private @Unknown1(%arg0: tensor<2x128xi64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<30522> : tensor<256xi64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<256xf64>
    %2 = mhlo.constant dense<0> : tensor<256xi64>
    %3 = "mhlo.reshape"(%arg0) : (tensor<2x128xi64>) -> tensor<256xi64>
    %4 = "mhlo.convert"(%3) : (tensor<256xi64>) -> tensor<256xui32>
    %5 = "mhlo.compare"(%3, %2) {comparison_direction = "LT"} : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    %6 = mhlo.add %3, %0 : tensor<256xi64>
    %7 = "mhlo.select"(%5, %6, %3) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %8 = "mhlo.reshape"(%7) : (tensor<256xi64>) -> tensor<256x1xi64>
    %9 = "mhlo.convert"(%3) : (tensor<256xi64>) -> tensor<256xf64>
    %10 = "mhlo.compare"(%9, %1) {comparison_direction = "NE"} : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
    return %4, %8, %10 : tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>
  }
  func private @Unknown2(%arg0: tensor<128xi64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0> : tensor<256xi64>
    %1 = mhlo.constant dense<2> : tensor<256xi64>
    %2 = mhlo.constant dense<-1.000000e+00> : tensor<256xf64>
    %3 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xi64>) -> tensor<2x128xi64>
    %4 = "mhlo.reshape"(%3) : (tensor<2x128xi64>) -> tensor<256xi64>
    %5 = "mhlo.convert"(%4) : (tensor<256xi64>) -> tensor<256xui32>
    %6 = "mhlo.compare"(%4, %0) {comparison_direction = "LT"} : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    %7 = mhlo.add %4, %1 : tensor<256xi64>
    %8 = "mhlo.select"(%6, %7, %4) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %9 = "mhlo.reshape"(%8) : (tensor<256xi64>) -> tensor<256x1xi64>
    %10 = "mhlo.convert"(%4) : (tensor<256xi64>) -> tensor<256xf64>
    %11 = "mhlo.compare"(%10, %2) {comparison_direction = "NE"} : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
    return %5, %9, %11 : tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>
  }
  func private @Unknown3(%arg0: tensor<1x128xi64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0> : tensor<128xi64>
    %1 = mhlo.constant dense<512> : tensor<128xi64>
    %2 = mhlo.constant dense<-1.000000e+00> : tensor<128xf64>
    %3 = "mhlo.reshape"(%arg0) : (tensor<1x128xi64>) -> tensor<128xi64>
    %4 = "mhlo.convert"(%3) : (tensor<128xi64>) -> tensor<128xui32>
    %5 = "mhlo.compare"(%3, %0) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %6 = mhlo.add %3, %1 : tensor<128xi64>
    %7 = "mhlo.select"(%5, %6, %3) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %8 = "mhlo.reshape"(%7) : (tensor<128xi64>) -> tensor<128x1xi64>
    %9 = "mhlo.convert"(%3) : (tensor<128xi64>) -> tensor<128xf64>
    %10 = "mhlo.compare"(%9, %2) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    return %4, %8, %10 : tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>
  }
  func private @Unknown4(%arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.reshape"(%arg0) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %1 = "mhlo.reshape"(%arg1) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %2 = mhlo.add %0, %1 : tensor<2x128x128xf32>
    %3 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<2x128x128xf32>
    %4 = mhlo.add %2, %3 : tensor<2x128x128xf32>
    return %4 : tensor<2x128x128xf32>
  }
  func private @Unknown5(%arg0: tensor<256x30522xf32>, %arg1: tensor<30522xf32>) -> (tensor<2x128x30522xf32>, tensor<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.reshape"(%arg0) : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<30522xf32>) -> tensor<2x128x30522xf32>
    %2 = mhlo.add %0, %1 : tensor<2x128x30522xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<2x128x30522xf32>) -> tensor<256x30522xf32>
    return %2, %3 : tensor<2x128x30522xf32>, tensor<256x30522xf32>
  }
  func private @Unknown6(%arg0: tensor<256xf32>, %arg1: tensor<256x30522xf32>) -> (tensor<256x30522xf32>, tensor<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<256x30522xf32>
    %1 = mhlo.subtract %arg1, %0 : tensor<256x30522xf32>
    %2 = "mhlo.exponential"(%1) : (tensor<256x30522xf32>) -> tensor<256x30522xf32>
    return %1, %2 : tensor<256x30522xf32>, tensor<256x30522xf32>
  }
  func private @Unknown7(%arg0: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.log"(%arg0) : (tensor<256xf32>) -> tensor<256xf32>
    return %0 : tensor<256xf32>
  }
  func private @Unknown8(%arg0: tensor<256xf32>, %arg1: tensor<256x30522xf32>, %arg2: tensor<256xi64>, %arg3: tensor<256xi1>) -> (tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<256x30522xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<256x30522xf32>
    %2 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<256x30522xf32>
    %3 = mhlo.subtract %arg1, %2 : tensor<256x30522xf32>
    %4 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi64>) -> tensor<256x30522xi64>
    %5 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<30522xi64>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<30522xi64>) -> tensor<256x30522xi64>
    %7 = "mhlo.compare"(%4, %6) {comparison_direction = "EQ"} : (tensor<256x30522xi64>, tensor<256x30522xi64>) -> tensor<256x30522xi1>
    %8 = "mhlo.select"(%7, %1, %0) : (tensor<256x30522xi1>, tensor<256x30522xf32>, tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %9 = "mhlo.compare"(%8, %1) {comparison_direction = "NE"} : (tensor<256x30522xf32>, tensor<256x30522xf32>) -> tensor<256x30522xi1>
    %10 = "mhlo.negate"(%8) : (tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %11 = mhlo.multiply %10, %3 : tensor<256x30522xf32>
    %12 = "mhlo.select"(%9, %0, %11) : (tensor<256x30522xi1>, tensor<256x30522xf32>, tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %13 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x30522xi1>
    %14 = "mhlo.select"(%13, %1, %0) : (tensor<256x30522xi1>, tensor<256x30522xf32>, tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %15 = mhlo.multiply %14, %8 : tensor<256x30522xf32>
    %16 = mhlo.multiply %12, %15 : tensor<256x30522xf32>
    %17 = mhlo.multiply %10, %15 : tensor<256x30522xf32>
    %18 = "mhlo.exponential"(%3) : (tensor<256x30522xf32>) -> tensor<256x30522xf32>
    return %15, %16, %17, %18 : tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>
  }
  func private @Unknown9(%arg0: tensor<f32>) -> tensor<f32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = "mhlo.compare"(%arg0, %1) {comparison_direction = "NE"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = "mhlo.select"(%2, %arg0, %0) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %3 : tensor<f32>
  }
  func private @Unknown10(%arg0: tensor<f32>, %arg1: tensor<256x30522xf32>) -> tensor<256x30522xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x30522xf32>
    %1 = mhlo.divide %arg1, %0 : tensor<256x30522xf32>
    return %1 : tensor<256x30522xf32>
  }
  func private @Unknown11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.divide %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func private @Unknown12(%arg0: tensor<256xf32>, %arg1: tensor<256x30522xf32>, %arg2: tensor<256x30522xf32>) -> (tensor<256x30522xf32>, tensor<2x128x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<256x30522xf32>
    %1 = mhlo.multiply %arg1, %0 : tensor<256x30522xf32>
    %2 = mhlo.subtract %arg2, %1 : tensor<256x30522xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    return %2, %3 : tensor<256x30522xf32>, tensor<2x128x30522xf32>
  }
  func private @Unknown13(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func private @Unknown14(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<2x128x128xf32>, %arg3: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x128x128xf32>
    %1 = mhlo.add %0, %arg2 : tensor<2x128x128xf32>
    %2 = mhlo.add %1, %arg3 : tensor<2x128x128xf32>
    return %2 : tensor<2x128x128xf32>
  }
  func private @Unknown15(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func private @Unknown16(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<2x128x128xf32>, %arg3: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x128x128xf32>
    %1 = mhlo.add %0, %arg2 : tensor<2x128x128xf32>
    %2 = mhlo.add %1, %arg3 : tensor<2x128x128xf32>
    return %2 : tensor<2x128x128xf32>
  }
  func private @Unknown17(%arg0: tensor<256xi1>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<256xi1>) -> (tensor<256x128xf32>, tensor<256x128xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<256x128xf32>
    %1 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x128xi1>
    %2 = "mhlo.reshape"(%arg1) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %3 = "mhlo.select"(%1, %2, %0) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %4 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x128xi1>
    %5 = "mhlo.select"(%4, %2, %0) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    return %3, %5 : tensor<256x128xf32>, tensor<256x128xf32>
  }
  func private @Unknown18(%arg0: tensor<128xi1>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<128x128xf32>
    %1 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %2 = "mhlo.select"(%1, %arg1, %0) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %2 : tensor<128x128xf32>
  }
  func @main(%arg0: tensor<2x128xi64>, %arg1: tensor<2x128xi64>, %arg2: tensor<1x512xi64>, %arg3: tensor<1x512xi64>, %arg4: tensor<30522x128xf32>, %arg5: tensor<2x128xf32>, %arg6: tensor<512x128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<128x128xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128x128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128x128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128x128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<512x128xf32>, %arg20: tensor<512xf32>, %arg21: tensor<128x512xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128x128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128x128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128x128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128x128xf32>, %arg32: tensor<128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<512x128xf32>, %arg36: tensor<512xf32>, %arg37: tensor<128x512xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128x128xf32>, %arg42: tensor<128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<30522xf32>) -> (tensor<2x128x30522xf32>, tensor<f32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>) {
    %0 = mhlo.constant dense<-0.000000e+00> : tensor<2x128x128xf32>
    %1 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<2x128xf32>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<512x128xf32>
    %4 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = "mhlo.slice"(%arg2) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %6 = "mhlo.reshape"(%5) : (tensor<1x128xi64>) -> tensor<128xi64>
    %7 = "mhlo.slice"(%arg3) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %8:2 = call @Unknown0(%arg1) : (tensor<2x128xi64>) -> (tensor<256xi64>, tensor<256xi1>)
    %9:3 = call @Unknown1(%arg0) : (tensor<2x128xi64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>)
    %10 = "mhlo.gather"(%arg4, %9#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    %11:3 = call @Unknown2(%6) : (tensor<128xi64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>)
    %12 = "mhlo.gather"(%arg5, %11#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<2x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    %13:3 = call @Unknown3(%7) : (tensor<1x128xi64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>)
    %14 = "mhlo.gather"(%arg6, %13#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<512x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %15 = call @Unknown4(%10, %12, %14) : (tensor<256x128xf32>, tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<2x128x128xf32>
    %16 = "mhlo.custom_call"(%15, %arg7, %arg8) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>
    %17 = "mhlo.get_tuple_element"(%16) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<2x128x128xf32>
    %18 = "mhlo.custom_call"(%17, %arg9, %arg10) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %19 = "mhlo.custom_call"(%17, %arg11, %arg12) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %20 = "mhlo.custom_call"(%18, %19) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %21 = "mhlo.custom_call"(%20, %0) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %22 = "mhlo.get_tuple_element"(%21) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %23 = "mhlo.custom_call"(%17, %arg13, %arg14) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %24 = "mhlo.custom_call"(%22, %23) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %25 = "mhlo.custom_call"(%24) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %26 = "mhlo.reshape"(%25) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %27 = "mhlo.custom_call"(%26, %arg15, %arg16) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x128x128xf32>
    %28 = "mhlo.custom_call"(%27, %arg17, %arg18, %17) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %29 = "mhlo.get_tuple_element"(%28) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %30 = "mhlo.custom_call"(%29, %arg19, %arg20) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>
    %31 = "mhlo.get_tuple_element"(%30) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %32 = "mhlo.custom_call"(%31, %arg21, %arg22) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tensor<2x128x128xf32>
    %33 = "mhlo.custom_call"(%32, %arg23, %arg24, %29) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %34 = "mhlo.get_tuple_element"(%33) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %35 = "mhlo.custom_call"(%34, %arg25, %arg26) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %36 = "mhlo.custom_call"(%34, %arg27, %arg28) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %37 = "mhlo.custom_call"(%35, %36) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %38 = "mhlo.custom_call"(%37, %0) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %39 = "mhlo.get_tuple_element"(%38) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %40 = "mhlo.custom_call"(%34, %arg29, %arg30) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %41 = "mhlo.custom_call"(%39, %40) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %42 = "mhlo.custom_call"(%41) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %43 = "mhlo.reshape"(%42) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %44 = "mhlo.custom_call"(%43, %arg31, %arg32) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x128x128xf32>
    %45 = "mhlo.custom_call"(%44, %arg33, %arg34, %34) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %46 = "mhlo.get_tuple_element"(%45) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %47 = "mhlo.custom_call"(%46, %arg35, %arg36) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>
    %48 = "mhlo.get_tuple_element"(%47) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %49 = "mhlo.custom_call"(%48, %arg37, %arg38) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tensor<2x128x128xf32>
    %50 = "mhlo.custom_call"(%49, %arg39, %arg40, %46) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %51 = "mhlo.get_tuple_element"(%50) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %52 = "mhlo.custom_call"(%51, %arg41, %arg42) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>
    %53 = "mhlo.get_tuple_element"(%52) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %54 = "mhlo.custom_call"(%53, %arg43, %arg44) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>
    %55 = "mhlo.get_tuple_element"(%54) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<2x128x128xf32>
    %56 = "mhlo.reshape"(%55) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %57 = "mhlo.dot_general"(%56, %arg4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256x128xf32>, tensor<30522x128xf32>) -> tensor<256x30522xf32>
    %58:2 = call @Unknown5(%57, %arg45) : (tensor<256x30522xf32>, tensor<30522xf32>) -> (tensor<2x128x30522xf32>, tensor<256x30522xf32>)
    %59 = mhlo.reduce(%58#1 init: %1) across dimensions = [1] : (tensor<256x30522xf32>, tensor<f32>) -> tensor<256xf32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %208 = mhlo.maximum %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }
    %60:2 = call @Unknown6(%59, %58#1) : (tensor<256xf32>, tensor<256x30522xf32>) -> (tensor<256x30522xf32>, tensor<256x30522xf32>)
    %61 = mhlo.reduce(%60#1 init: %4) across dimensions = [1] : (tensor<256x30522xf32>, tensor<f32>) -> tensor<256xf32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %208 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }
    %62 = call @Unknown7(%61) : (tensor<256xf32>) -> tensor<256xf32>
    %63:4 = call @Unknown8(%62, %60#0, %8#0, %8#1) : (tensor<256xf32>, tensor<256x30522xf32>, tensor<256xi64>, tensor<256xi1>) -> (tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>)
    %64 = mhlo.reduce(%63#0 init: %4) across dimensions = [0, 1] : (tensor<256x30522xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %208 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }
    %65 = mhlo.reduce(%63#0 init: %4) across dimensions = [0, 1] : (tensor<256x30522xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %208 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }
    %66 = call @Unknown9(%65) : (tensor<f32>) -> tensor<f32>
    %67 = call @Unknown10(%66, %63#2) : (tensor<f32>, tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %68 = mhlo.reduce(%67 init: %4) across dimensions = [1] : (tensor<256x30522xf32>, tensor<f32>) -> tensor<256xf32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %208 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }
    %69 = mhlo.reduce(%63#1 init: %4) across dimensions = [0, 1] : (tensor<256x30522xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %208 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }
    %70 = call @Unknown11(%69, %64) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %71 = "mhlo.get_tuple_element"(%54) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %72 = "mhlo.get_tuple_element"(%54) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %73 = "mhlo.get_tuple_element"(%52) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %74 = "mhlo.get_tuple_element"(%52) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %75 = "mhlo.get_tuple_element"(%50) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %76 = "mhlo.get_tuple_element"(%50) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %77 = "mhlo.get_tuple_element"(%50) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %78 = "mhlo.get_tuple_element"(%47) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %79 = "mhlo.get_tuple_element"(%47) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %80 = "mhlo.get_tuple_element"(%45) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %81 = "mhlo.get_tuple_element"(%45) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %82 = "mhlo.get_tuple_element"(%45) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %83 = "mhlo.get_tuple_element"(%38) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %84 = "mhlo.get_tuple_element"(%33) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %85 = "mhlo.get_tuple_element"(%33) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %86 = "mhlo.get_tuple_element"(%33) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %87 = "mhlo.get_tuple_element"(%30) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %88 = "mhlo.get_tuple_element"(%30) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %89 = "mhlo.get_tuple_element"(%28) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %90 = "mhlo.get_tuple_element"(%28) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %91 = "mhlo.get_tuple_element"(%28) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %92 = "mhlo.get_tuple_element"(%21) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %93 = "mhlo.get_tuple_element"(%16) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %94 = "mhlo.get_tuple_element"(%16) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %95:2 = call @Unknown12(%68, %63#3, %67) : (tensor<256xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>) -> (tensor<256x30522xf32>, tensor<2x128x30522xf32>)
    %96 = call @MatmulOp0(%56, %95#0) : (tensor<256x128xf32>, tensor<256x30522xf32>) -> tensor<30522x128xf32>
    %97 = "mhlo.dot"(%95#0, %arg4) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256x30522xf32>, tensor<30522x128xf32>) -> tensor<256x128xf32>
    %98 = "mhlo.reshape"(%97) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %99 = "mhlo.custom_call"(%98, %53, %arg43, %71, %72) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %100 = "mhlo.get_tuple_element"(%99) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %101 = "mhlo.custom_call"(%100, %51, %arg41, %73, %74) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %102 = "mhlo.get_tuple_element"(%101) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %103 = "mhlo.custom_call"(%102, %75, %arg39, %76, %77) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %104 = "mhlo.get_tuple_element"(%103) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %105 = "mhlo.get_tuple_element"(%103) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %106 = "mhlo.custom_call"(%105, %48, %arg37) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %107 = "mhlo.get_tuple_element"(%106) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %108 = "mhlo.custom_call"(%107, %46, %arg35, %78, %79) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %109 = "mhlo.get_tuple_element"(%108) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %110 = call @Unknown13(%104, %109) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %111 = "mhlo.custom_call"(%110, %80, %arg33, %81, %82) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %112 = "mhlo.get_tuple_element"(%111) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %113 = "mhlo.get_tuple_element"(%111) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %114 = "mhlo.custom_call"(%113, %43, %arg31) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %115 = "mhlo.get_tuple_element"(%114) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %116 = "mhlo.reshape"(%115) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %117 = "mhlo.custom_call"(%116) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %118 = "mhlo.custom_call"(%117, %39, %40) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %119 = "mhlo.get_tuple_element"(%118) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %120 = "mhlo.custom_call"(%119, %39, %83) {api_version = 1 : i32, backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %121 = "mhlo.custom_call"(%120, %35, %36) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %122 = "mhlo.get_tuple_element"(%121) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %123 = "mhlo.custom_call"(%122, %34, %arg25) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %124 = "mhlo.get_tuple_element"(%123) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %125 = "mhlo.get_tuple_element"(%118) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %126 = "mhlo.custom_call"(%125, %34, %arg29) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %127 = "mhlo.get_tuple_element"(%126) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %128 = "mhlo.get_tuple_element"(%121) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %129 = "mhlo.custom_call"(%128, %34, %arg27) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %130 = "mhlo.get_tuple_element"(%129) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %131 = call @Unknown14(%112, %124, %127, %130) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %132 = "mhlo.custom_call"(%131, %84, %arg23, %85, %86) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %133 = "mhlo.get_tuple_element"(%132) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %134 = "mhlo.get_tuple_element"(%132) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %135 = "mhlo.custom_call"(%134, %31, %arg21) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %136 = "mhlo.get_tuple_element"(%135) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %137 = "mhlo.custom_call"(%136, %29, %arg19, %87, %88) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %138 = "mhlo.get_tuple_element"(%137) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %139 = call @Unknown15(%133, %138) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %140 = "mhlo.custom_call"(%139, %89, %arg17, %90, %91) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %141 = "mhlo.get_tuple_element"(%140) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %142 = "mhlo.get_tuple_element"(%140) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %143 = "mhlo.custom_call"(%142, %26, %arg15) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %144 = "mhlo.get_tuple_element"(%143) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %145 = "mhlo.reshape"(%144) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %146 = "mhlo.custom_call"(%145) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %147 = "mhlo.custom_call"(%146, %22, %23) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %148 = "mhlo.get_tuple_element"(%147) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %149 = "mhlo.custom_call"(%148, %22, %92) {api_version = 1 : i32, backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %150 = "mhlo.custom_call"(%149, %18, %19) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %151 = "mhlo.get_tuple_element"(%150) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %152 = "mhlo.custom_call"(%151, %17, %arg9) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %153 = "mhlo.get_tuple_element"(%152) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %154 = "mhlo.get_tuple_element"(%147) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %155 = "mhlo.custom_call"(%154, %17, %arg13) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %156 = "mhlo.get_tuple_element"(%155) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %157 = "mhlo.get_tuple_element"(%150) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %158 = "mhlo.custom_call"(%157, %17, %arg11) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %159 = "mhlo.get_tuple_element"(%158) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %160 = call @Unknown16(%141, %153, %156, %159) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %161 = "mhlo.custom_call"(%160, %15, %arg7, %93, %94) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %162 = "mhlo.get_tuple_element"(%161) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %163:2 = call @Unknown17(%9#2, %162, %11#2) : (tensor<256xi1>, tensor<2x128x128xf32>, tensor<256xi1>) -> (tensor<256x128xf32>, tensor<256x128xf32>)
    %164 = "mhlo.scatter"(%96, %9#1, %163#0) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):  // no predecessors
      %208 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    %165 = "mhlo.scatter"(%2, %11#1, %163#1) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):  // no predecessors
      %208 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<2x128xf32>
    %166 = mhlo.reduce(%162 init: %4) across dimensions = [0] : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %208 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }
    %167 = call @Unknown18(%13#2, %166) : (tensor<128xi1>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %168 = "mhlo.scatter"(%3, %13#1, %167) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):  // no predecessors
      %208 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %169 = "mhlo.get_tuple_element"(%161) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %170 = "mhlo.get_tuple_element"(%161) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %171 = "mhlo.get_tuple_element"(%152) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %172 = "mhlo.get_tuple_element"(%152) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %173 = "mhlo.get_tuple_element"(%158) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %174 = "mhlo.get_tuple_element"(%158) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %175 = "mhlo.get_tuple_element"(%155) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %176 = "mhlo.get_tuple_element"(%155) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %177 = "mhlo.get_tuple_element"(%143) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %178 = "mhlo.get_tuple_element"(%143) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %179 = "mhlo.get_tuple_element"(%140) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %180 = "mhlo.get_tuple_element"(%140) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %181 = "mhlo.get_tuple_element"(%137) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %182 = "mhlo.get_tuple_element"(%137) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %183 = "mhlo.get_tuple_element"(%135) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %184 = "mhlo.get_tuple_element"(%135) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %185 = "mhlo.get_tuple_element"(%132) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %186 = "mhlo.get_tuple_element"(%132) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %187 = "mhlo.get_tuple_element"(%123) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %188 = "mhlo.get_tuple_element"(%123) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %189 = "mhlo.get_tuple_element"(%129) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %190 = "mhlo.get_tuple_element"(%129) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %191 = "mhlo.get_tuple_element"(%126) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %192 = "mhlo.get_tuple_element"(%126) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %193 = "mhlo.get_tuple_element"(%114) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %194 = "mhlo.get_tuple_element"(%114) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %195 = "mhlo.get_tuple_element"(%111) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %196 = "mhlo.get_tuple_element"(%111) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %197 = "mhlo.get_tuple_element"(%108) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %198 = "mhlo.get_tuple_element"(%108) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %199 = "mhlo.get_tuple_element"(%106) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %200 = "mhlo.get_tuple_element"(%106) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %201 = "mhlo.get_tuple_element"(%103) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %202 = "mhlo.get_tuple_element"(%103) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %203 = "mhlo.get_tuple_element"(%101) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %204 = "mhlo.get_tuple_element"(%101) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %205 = "mhlo.get_tuple_element"(%99) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %206 = "mhlo.get_tuple_element"(%99) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %207 = mhlo.reduce(%95#1 init: %4) across dimensions = [0, 1] : (tensor<2x128x30522xf32>, tensor<f32>) -> tensor<30522xf32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %208 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%208) : (tensor<f32>) -> ()
    }
    return %58#0, %70, %164, %165, %168, %169, %170, %171, %172, %173, %174, %175, %176, %177, %178, %179, %180, %181, %182, %183, %184, %185, %186, %187, %188, %189, %190, %191, %192, %193, %194, %195, %196, %197, %198, %199, %200, %201, %202, %203, %204, %205, %206, %207 : tensor<2x128x30522xf32>, tensor<f32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>
  }
}

