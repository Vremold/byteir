// RUN: byteir-opt %s -hlo-fusion-to-linalg="anchor-tag=byre_elementwise_fusion" -unrealized-cast-to-linalg -linalg-fuse-elementwise-ops -cse | FileCheck %s

module  {
  func private @MatmulOp0(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
  func private @MatmulOp1(%arg0: tensor<128x128xf32>, %arg1: tensor<128x30522xf32>) -> tensor<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x30522xf32>) -> tensor<128x30522xf32>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %1 : tensor<30522x128xf32>
  }
  func private @Unknown0(%arg0: tensor<1x128xi64>, %arg1: tensor<128xi64>, %arg2: tensor<128xi64>, %arg3: tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>) attributes {byre_elementwise_fusion} {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x128xi64>) -> tensor<128xi64>
    %1 = "mhlo.convert"(%0) : (tensor<128xi64>) -> tensor<128xui32>
    %2 = "mhlo.compare"(%0, %arg1) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %3 = mhlo.add %0, %arg2 : tensor<128xi64>
    %4 = "mhlo.select"(%2, %3, %0) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %5 = "mhlo.reshape"(%4) : (tensor<128xi64>) -> tensor<128x1xi64>
    %6 = "mhlo.convert"(%0) : (tensor<128xi64>) -> tensor<128xf64>
    %7 = "mhlo.compare"(%6, %arg3) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    return %1, %5, %7 : tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>
  }
  func private @Unknown1(%arg0: tensor<1x128xi64>, %arg1: tensor<128xi64>, %arg2: tensor<128xi64>, %arg3: tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>) attributes {byre_elementwise_fusion} {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x128xi64>) -> tensor<128xi64>
    %1 = "mhlo.convert"(%0) : (tensor<128xi64>) -> tensor<128xui32>
    %2 = "mhlo.compare"(%0, %arg1) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %3 = mhlo.add %0, %arg2 : tensor<128xi64>
    %4 = "mhlo.select"(%2, %3, %0) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %5 = "mhlo.reshape"(%4) : (tensor<128xi64>) -> tensor<128x1xi64>
    %6 = "mhlo.convert"(%0) : (tensor<128xi64>) -> tensor<128xf64>
    %7 = "mhlo.compare"(%6, %arg3) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    return %1, %5, %7 : tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>
  }
  func private @Unknown2(%arg0: tensor<1x128xi64>, %arg1: tensor<128xi64>, %arg2: tensor<128xi64>, %arg3: tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>) attributes {byre_elementwise_fusion} {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x128xi64>) -> tensor<128xi64>
    %1 = "mhlo.convert"(%0) : (tensor<128xi64>) -> tensor<128xui32>
    %2 = "mhlo.compare"(%0, %arg1) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %3 = mhlo.add %0, %arg2 : tensor<128xi64>
    %4 = "mhlo.select"(%2, %3, %0) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %5 = "mhlo.reshape"(%4) : (tensor<128xi64>) -> tensor<128x1xi64>
    %6 = "mhlo.convert"(%0) : (tensor<128xi64>) -> tensor<128xf64>
    %7 = "mhlo.compare"(%6, %arg3) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    return %1, %5, %7 : tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>
  }
  func private @Unknown3(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = "mhlo.reshape"(%arg0) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %1 = "mhlo.reshape"(%arg1) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %2 = mhlo.add %0, %1 : tensor<1x128x128xf32>
    %3 = "mhlo.reshape"(%arg2) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %4 = mhlo.add %2, %3 : tensor<1x128x128xf32>
    %5 = "mhlo.reshape"(%4) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    return %5 : tensor<128x128xf32>
  }
  func private @Unknown4(%arg0: tensor<128x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x128x128xf32>, %arg4: tensor<1x128x128xf32>, %arg5: tensor<1x128x128xf32>, %arg6: tensor<1x128x128xf32>, %arg7: tensor<1x128x128xf32>, %arg8: tensor<1x128x128xf32>, %arg9: tensor<1x128x128xf32>, %arg10: tensor<1x128x128xf32>, %arg11: tensor<1x128x128xf32>, %arg12: tensor<1x128x128xf32>, %arg13: tensor<1x128x128xf32>, %arg14: tensor<1x128x128xf32>, %arg15: tensor<1x128x128xf32>, %arg16: tensor<1x128x128xf32>, %arg17: tensor<1x128x128xf32>, %arg18: tensor<1x128x128xf32>, %arg19: tensor<1x128x128xf32>, %arg20: tensor<1x128x128xf32>, %arg21: tensor<1x128x128xf32>) -> (tensor<1x128x128xf32>, tensor<1x128x128xf32>) attributes {byre_elementwise_fusion} {
    %0 = "mhlo.reshape"(%arg0) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %2 = mhlo.add %0, %1 : tensor<1x128x128xf32>
    %3 = mhlo.multiply %2, %arg2 : tensor<1x128x128xf32>
    %4 = mhlo.multiply %2, %arg3 : tensor<1x128x128xf32>
    %5 = "mhlo.clamp"(%arg4, %4, %arg5) : (tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    %6 = mhlo.multiply %5, %5 : tensor<1x128x128xf32>
    %7 = mhlo.multiply %6, %arg6 : tensor<1x128x128xf32>
    %8 = mhlo.add %7, %arg7 : tensor<1x128x128xf32>
    %9 = mhlo.multiply %8, %6 : tensor<1x128x128xf32>
    %10 = mhlo.add %9, %arg8 : tensor<1x128x128xf32>
    %11 = mhlo.multiply %10, %6 : tensor<1x128x128xf32>
    %12 = mhlo.add %11, %arg9 : tensor<1x128x128xf32>
    %13 = mhlo.multiply %12, %6 : tensor<1x128x128xf32>
    %14 = mhlo.add %13, %arg10 : tensor<1x128x128xf32>
    %15 = mhlo.multiply %14, %6 : tensor<1x128x128xf32>
    %16 = mhlo.add %15, %arg11 : tensor<1x128x128xf32>
    %17 = mhlo.multiply %16, %6 : tensor<1x128x128xf32>
    %18 = mhlo.add %17, %arg12 : tensor<1x128x128xf32>
    %19 = mhlo.multiply %18, %6 : tensor<1x128x128xf32>
    %20 = mhlo.add %19, %arg13 : tensor<1x128x128xf32>
    %21 = mhlo.multiply %5, %20 : tensor<1x128x128xf32>
    %22 = mhlo.add %7, %arg14 : tensor<1x128x128xf32>
    %23 = mhlo.multiply %22, %6 : tensor<1x128x128xf32>
    %24 = mhlo.add %23, %arg15 : tensor<1x128x128xf32>
    %25 = mhlo.multiply %24, %6 : tensor<1x128x128xf32>
    %26 = mhlo.add %25, %arg16 : tensor<1x128x128xf32>
    %27 = mhlo.multiply %26, %6 : tensor<1x128x128xf32>
    %28 = mhlo.add %27, %arg17 : tensor<1x128x128xf32>
    %29 = mhlo.multiply %28, %6 : tensor<1x128x128xf32>
    %30 = mhlo.add %29, %arg18 : tensor<1x128x128xf32>
    %31 = mhlo.divide %21, %30 : tensor<1x128x128xf32>
    %32 = mhlo.add %31, %arg19 : tensor<1x128x128xf32>
    %33 = mhlo.multiply %3, %32 : tensor<1x128x128xf32>
    %34 = mhlo.multiply %32, %arg2 : tensor<1x128x128xf32>
    %35 = mhlo.multiply %2, %2 : tensor<1x128x128xf32>
    %36 = mhlo.multiply %35, %arg20 : tensor<1x128x128xf32>
    %37 = "mhlo.exponential"(%36) : (tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    %38 = mhlo.multiply %2, %37 : tensor<1x128x128xf32>
    %39 = mhlo.multiply %38, %arg21 : tensor<1x128x128xf32>
    %40 = mhlo.add %34, %39 : tensor<1x128x128xf32>
    return %33, %40 : tensor<1x128x128xf32>, tensor<1x128x128xf32>
  }
  func private @Unknown5(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<128xf32> attributes {byre_elementwise_fusion} {
    %0 = mhlo.add %arg0, %arg1 : tensor<128xf32>
    %1 = "mhlo.rsqrt"(%0) : (tensor<128xf32>) -> tensor<128xf32>
    %2 = mhlo.divide %arg2, %1 : tensor<128xf32>
    %3 = mhlo.multiply %2, %2 : tensor<128xf32>
    %4 = mhlo.subtract %3, %arg1 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown6(%arg0: tensor<128x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<128xf32>) -> (tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<128x128xf32>, tensor<1x128x128xf32>) attributes {byre_elementwise_fusion} {
    %0 = "mhlo.reshape"(%arg0) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %2 = mhlo.multiply %0, %1 : tensor<1x128x128xf32>
    %3 = mhlo.multiply %arg2, %1 : tensor<1x128x128xf32>
    %4 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %5 = mhlo.add %4, %3 : tensor<1x128x128xf32>
    %6 = "mhlo.reshape"(%5) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %7 = mhlo.multiply %0, %arg2 : tensor<1x128x128xf32>
    return %0, %2, %6, %7 : tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<128x128xf32>, tensor<1x128x128xf32>
  }
  func private @Unknown7(%arg0: tensor<128x30522xf32>, %arg1: tensor<30522xf32>) -> tensor<1x128x30522xf32> attributes {byre_elementwise_fusion} {
    %0 = "mhlo.reshape"(%arg0) : (tensor<128x30522xf32>) -> tensor<1x128x30522xf32>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<30522xf32>) -> tensor<1x128x30522xf32>
    %2 = mhlo.add %0, %1 : tensor<1x128x30522xf32>
    return %2 : tensor<1x128x30522xf32>
  }
  func private @Unknown8(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x128xf32>) -> (tensor<1x128x128xf32>, tensor<128x128xf32>) attributes {byre_elementwise_fusion} {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<1x128x128xf32>
    %1 = "mhlo.reshape"(%0) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    return %0, %1 : tensor<1x128x128xf32>, tensor<128x128xf32>
  }
  func private @Unknown9(%arg0: tensor<128xi1>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %1 = "mhlo.select"(%0, %arg1, %arg2) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
  func private @Unknown10(%arg0: tensor<128xi1>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %1 = "mhlo.select"(%0, %arg1, %arg2) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
  func private @Unknown11(%arg0: tensor<128xi1>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %1 = "mhlo.select"(%0, %arg1, %arg2) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
  func @main(%arg0: tensor<30522x128xf32>, %arg1: tensor<512x128xf32>, %arg2: tensor<2x128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128x128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128x128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<128x128xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128x128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<512x128xf32>, %arg16: tensor<512xf32>, %arg17: tensor<128x512xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<128x128xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128x128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128x128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128x128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<512x128xf32>, %arg32: tensor<512xf32>, %arg33: tensor<128x512xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128xf32>, %arg37: tensor<30522xf32>, %arg38: tensor<128x128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<30522x128xf32>, %arg43: tensor<30522xf32>, %arg44: tensor<1x512xi64>, %arg45: tensor<1x512xi64>, %arg46: tensor<1x128xi64>) -> (tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>) {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<1x128x128xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<128x128xf32>
    %3 = mhlo.constant dense<-1.000000e+00> : tensor<128xf64>
    %4 = mhlo.constant dense<2> : tensor<128xi64>
    %5 = mhlo.constant dense<0> : tensor<128xi64>
    %6 = mhlo.constant dense<0.000000e+00> : tensor<2x128xf32>
    %7 = mhlo.constant dense<512> : tensor<128xi64>
    %8 = mhlo.constant dense<0.000000e+00> : tensor<512x128xf32>
    %9 = mhlo.constant dense<-4.000000e+00> : tensor<1x128x128xf32>
    %10 = mhlo.constant dense<4.000000e+00> : tensor<1x128x128xf32>
    %11 = mhlo.constant dense<-2.72614237E-10> : tensor<1x128x128xf32>
    %12 = mhlo.constant dense<2.77068146E-8> : tensor<1x128x128xf32>
    %13 = mhlo.constant dense<-2.10102394E-6> : tensor<1x128x128xf32>
    %14 = mhlo.constant dense<-5.69250624E-5> : tensor<1x128x128xf32>
    %15 = mhlo.constant dense<-7.34990637E-4> : tensor<1x128x128xf32>
    %16 = mhlo.constant dense<-2.954600e-03> : tensor<1x128x128xf32>
    %17 = mhlo.constant dense<-0.0160960332> : tensor<1x128x128xf32>
    %18 = mhlo.constant dense<0.000000e+00> : tensor<1x128x128xf32>
    %19 = mhlo.constant dense<-1.45660715E-5> : tensor<1x128x128xf32>
    %20 = mhlo.constant dense<-2.13374049E-4> : tensor<1x128x128xf32>
    %21 = mhlo.constant dense<-0.00168282702> : tensor<1x128x128xf32>
    %22 = mhlo.constant dense<-0.00737332925> : tensor<1x128x128xf32>
    %23 = mhlo.constant dense<-0.0142647391> : tensor<1x128x128xf32>
    %24 = mhlo.constant dense<0.000000e+00> : tensor<128xf64>
    %25 = mhlo.constant dense<30522> : tensor<128xi64>
    %26 = mhlo.constant dense<0.000000e+00> : tensor<30522x128xf32>
    %27 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %28 = mhlo.constant dense<9.99999996E-13> : tensor<128xf32>
    %29 = mhlo.constant dense<1.000000e+00> : tensor<128x30522xf32>
    %30 = mhlo.constant dense<1.000000e+00> : tensor<1x128x30522xf32>
    %31 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %32 = mhlo.constant dense<5.000000e-01> : tensor<1x128x128xf32>
    %33 = mhlo.constant dense<0.707106769> : tensor<1x128x128xf32>
    %34 = mhlo.constant dense<-5.000000e-01> : tensor<1x128x128xf32>
    %35 = mhlo.constant dense<0.398942292> : tensor<1x128x128xf32>
    %36 = "mhlo.slice"(%arg45) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %37 = "mhlo.slice"(%arg44) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %38 = "mhlo.dot"(%29, %arg42) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x30522xf32>, tensor<30522x128xf32>) -> tensor<128x128xf32>
    %39:3 = call @Unknown0(%arg46, %5, %25, %24) : (tensor<1x128xi64>, tensor<128xi64>, tensor<128xi64>, tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>)
    %40 = "mhlo.gather"(%arg0, %39#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %41:3 = call @Unknown1(%37, %5, %7, %3) : (tensor<1x128xi64>, tensor<128xi64>, tensor<128xi64>, tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>)
    %42 = "mhlo.gather"(%arg1, %41#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<512x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %43:3 = call @Unknown2(%36, %5, %4, %3) : (tensor<1x128xi64>, tensor<128xi64>, tensor<128xi64>, tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>)
    %44 = "mhlo.gather"(%arg2, %43#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<2x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %45 = call @Unknown3(%40, %44, %42) : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %46 = "mhlo.dot_general"(%45, %arg38) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %47:2 = call @Unknown4(%46, %arg39, %32, %33, %9, %10, %18, %11, %12, %13, %14, %15, %16, %17, %19, %20, %21, %22, %23, %1, %34, %35) : (tensor<128x128xf32>, tensor<128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> (tensor<1x128x128xf32>, tensor<1x128x128xf32>)
    %48 = "mhlo.batch_norm_training"(%47#0, %27, %31) {epsilon = 9.99999996E-13 : f32, feature_index = 1 : i64} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %49 = "mhlo.get_tuple_element"(%48) {index = 0 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x128xf32>
    %50 = "mhlo.get_tuple_element"(%48) {index = 1 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %51 = "mhlo.get_tuple_element"(%48) {index = 2 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %52 = call @Unknown5(%51, %28, %27) : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %53:4 = call @Unknown6(%38, %arg40, %49, %arg41) : (tensor<128x128xf32>, tensor<128xf32>, tensor<1x128x128xf32>, tensor<128xf32>) -> (tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<128x128xf32>, tensor<1x128x128xf32>)
    %54 = "mhlo.batch_norm_grad"(%47#0, %27, %50, %52, %53#1) {epsilon = 9.99999996E-13 : f32, feature_index = 1 : i64} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x128xf32>) -> tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %55 = "mhlo.dot_general"(%53#2, %arg42) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<30522x128xf32>) -> tensor<128x30522xf32>
    %56 = call @Unknown7(%55, %arg43) : (tensor<128x30522xf32>, tensor<30522xf32>) -> tensor<1x128x30522xf32>
    %57 = "mhlo.get_tuple_element"(%54) {index = 0 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x128xf32>
    %58:2 = call @Unknown8(%57, %47#1) : (tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> (tensor<1x128x128xf32>, tensor<128x128xf32>)
    %59 = "mhlo.dot"(%58#1, %arg38) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %60 = call @Unknown9(%39#2, %59, %2) : (tensor<128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %61 = "mhlo.scatter"(%26, %39#1, %60) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %72 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%72) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<30522x128xf32>
    %62 = call @Unknown10(%41#2, %59, %2) : (tensor<128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %63 = "mhlo.scatter"(%8, %41#1, %62) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %72 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%72) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %64 = call @Unknown11(%43#2, %59, %2) : (tensor<128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %65 = "mhlo.scatter"(%6, %43#1, %64) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %72 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%72) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %66 = call @MatmulOp0(%45, %58#1) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %67 = "mhlo.reduce"(%58#0, %0) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %72 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%72) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %68 = "mhlo.reduce"(%53#3, %0) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %72 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%72) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %69 = "mhlo.reduce"(%53#0, %0) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %72 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%72) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %70 = call @MatmulOp1(%53#2, %29) : (tensor<128x128xf32>, tensor<128x30522xf32>) -> tensor<30522x128xf32>
    %71 = "mhlo.reduce"(%30, %0) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %72 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%72) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x30522xf32>, tensor<f32>) -> tensor<30522xf32>
    return %56, %61, %63, %65, %66, %67, %68, %69, %70, %71 : tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>
  }
}

// CHECK-LABEL: func private @MatmulOp0(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"}
// CHECK-NEXT: mhlo.dot_general

// CHECK-LABEL: func private @MatmulOp1(%arg0: tensor<128x128xf32>, %arg1: tensor<128x30522xf32>) -> tensor<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"}
// CHECK-NEXT: mhlo.dot_general

// CHECK-LABEL: func private @Unknown0
// CHECK-NEXT: linalg

// CHECK-LABEL: func private @Unknown1
// CHECK-NEXT: linalg

// CHECK-LABEL: func @main
