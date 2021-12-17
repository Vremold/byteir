// RUN: byteir-opt %s -fuse-element="attach-tag=byre_elementwise_fusion" -cse -expand-hlo-tuples="entry-function=main" -cse -mhlo-flatten-tuple -cse  -fusion-outlining -cse | FileCheck %s

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
  func @main(%arg0: tensor<30522x128xf32>, %arg1: tensor<512x128xf32>, %arg2: tensor<2x128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128x128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128x128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<128x128xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128x128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<512x128xf32>, %arg16: tensor<512xf32>, %arg17: tensor<128x512xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<128x128xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128x128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128x128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128x128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<512x128xf32>, %arg32: tensor<512xf32>, %arg33: tensor<128x512xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128xf32>, %arg37: tensor<30522xf32>, %arg38: tensor<128x128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<30522x128xf32>, %arg43: tensor<30522xf32>, %arg44: tensor<1x512xi64>, %arg45: tensor<1x512xi64>, %arg46: tensor<1x128xi64>) -> tuple<tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>> {
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
    %36 = "mhlo.reshape"(%arg46) : (tensor<1x128xi64>) -> tensor<128xi64>
    %37 = "mhlo.convert"(%36) : (tensor<128xi64>) -> tensor<128xui32>
    %38 = "mhlo.gather"(%arg0, %37) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %39 = "mhlo.reshape"(%38) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %40 = "mhlo.slice"(%arg45) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %41 = "mhlo.reshape"(%40) : (tensor<1x128xi64>) -> tensor<128xi64>
    %42 = "mhlo.convert"(%41) : (tensor<128xi64>) -> tensor<128xui32>
    %43 = "mhlo.gather"(%arg2, %42) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<2x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %44 = "mhlo.reshape"(%43) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %45 = mhlo.add %39, %44 : tensor<1x128x128xf32>
    %46 = "mhlo.slice"(%arg44) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %47 = "mhlo.reshape"(%46) : (tensor<1x128xi64>) -> tensor<128xi64>
    %48 = "mhlo.convert"(%47) : (tensor<128xi64>) -> tensor<128xui32>
    %49 = "mhlo.gather"(%arg1, %48) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<512x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %50 = "mhlo.reshape"(%49) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %51 = mhlo.add %45, %50 : tensor<1x128x128xf32>
    %52 = "mhlo.reshape"(%51) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %53 = "mhlo.dot_general"(%52, %arg38) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %54 = "mhlo.reshape"(%53) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %55 = "mhlo.broadcast_in_dim"(%arg39) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %56 = mhlo.add %54, %55 : tensor<1x128x128xf32>
    %57 = mhlo.multiply %56, %32 : tensor<1x128x128xf32>
    %58 = mhlo.multiply %56, %33 : tensor<1x128x128xf32>
    %59 = "mhlo.clamp"(%9, %58, %10) : (tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    %60 = mhlo.multiply %59, %59 : tensor<1x128x128xf32>
    %61 = mhlo.multiply %60, %18 : tensor<1x128x128xf32>
    %62 = mhlo.add %61, %11 : tensor<1x128x128xf32>
    %63 = mhlo.multiply %62, %60 : tensor<1x128x128xf32>
    %64 = mhlo.add %63, %12 : tensor<1x128x128xf32>
    %65 = mhlo.multiply %64, %60 : tensor<1x128x128xf32>
    %66 = mhlo.add %65, %13 : tensor<1x128x128xf32>
    %67 = mhlo.multiply %66, %60 : tensor<1x128x128xf32>
    %68 = mhlo.add %67, %14 : tensor<1x128x128xf32>
    %69 = mhlo.multiply %68, %60 : tensor<1x128x128xf32>
    %70 = mhlo.add %69, %15 : tensor<1x128x128xf32>
    %71 = mhlo.multiply %70, %60 : tensor<1x128x128xf32>
    %72 = mhlo.add %71, %16 : tensor<1x128x128xf32>
    %73 = mhlo.multiply %72, %60 : tensor<1x128x128xf32>
    %74 = mhlo.add %73, %17 : tensor<1x128x128xf32>
    %75 = mhlo.multiply %59, %74 : tensor<1x128x128xf32>
    %76 = mhlo.add %61, %19 : tensor<1x128x128xf32>
    %77 = mhlo.multiply %76, %60 : tensor<1x128x128xf32>
    %78 = mhlo.add %77, %20 : tensor<1x128x128xf32>
    %79 = mhlo.multiply %78, %60 : tensor<1x128x128xf32>
    %80 = mhlo.add %79, %21 : tensor<1x128x128xf32>
    %81 = mhlo.multiply %80, %60 : tensor<1x128x128xf32>
    %82 = mhlo.add %81, %22 : tensor<1x128x128xf32>
    %83 = mhlo.multiply %82, %60 : tensor<1x128x128xf32>
    %84 = mhlo.add %83, %23 : tensor<1x128x128xf32>
    %85 = mhlo.divide %75, %84 : tensor<1x128x128xf32>
    %86 = mhlo.add %85, %1 : tensor<1x128x128xf32>
    %87 = mhlo.multiply %57, %86 : tensor<1x128x128xf32>
    %88 = "mhlo.batch_norm_training"(%87, %27, %31) {epsilon = 9.99999996E-13 : f32, feature_index = 1 : i64} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %89 = "mhlo.get_tuple_element"(%88) {index = 0 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x128xf32>
    %90 = "mhlo.get_tuple_element"(%88) {index = 1 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %91 = "mhlo.get_tuple_element"(%88) {index = 2 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %92 = mhlo.add %91, %28 : tensor<128xf32>
    %93 = "mhlo.rsqrt"(%92) : (tensor<128xf32>) -> tensor<128xf32>
    %94 = "mhlo.dot"(%29, %arg42) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x30522xf32>, tensor<30522x128xf32>) -> tensor<128x128xf32>
    %95 = "mhlo.reshape"(%94) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %96 = "mhlo.broadcast_in_dim"(%arg40) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %97 = mhlo.multiply %95, %96 : tensor<1x128x128xf32>
    %98 = mhlo.divide %27, %93 : tensor<128xf32>
    %99 = mhlo.multiply %98, %98 : tensor<128xf32>
    %100 = mhlo.subtract %99, %28 : tensor<128xf32>
    %101 = "mhlo.batch_norm_grad"(%87, %27, %90, %100, %97) {epsilon = 9.99999996E-13 : f32, feature_index = 1 : i64} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x128xf32>) -> tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %102 = mhlo.multiply %89, %96 : tensor<1x128x128xf32>
    %103 = "mhlo.broadcast_in_dim"(%arg41) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %104 = mhlo.add %103, %102 : tensor<1x128x128xf32>
    %105 = "mhlo.reshape"(%104) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %106 = "mhlo.dot_general"(%105, %arg42) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<30522x128xf32>) -> tensor<128x30522xf32>
    %107 = "mhlo.reshape"(%106) : (tensor<128x30522xf32>) -> tensor<1x128x30522xf32>
    %108 = "mhlo.broadcast_in_dim"(%arg43) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<30522xf32>) -> tensor<1x128x30522xf32>
    %109 = mhlo.add %107, %108 : tensor<1x128x30522xf32>
    %110 = "mhlo.compare"(%36, %5) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %111 = mhlo.add %36, %25 : tensor<128xi64>
    %112 = "mhlo.select"(%110, %111, %36) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %113 = "mhlo.reshape"(%112) : (tensor<128xi64>) -> tensor<128x1xi64>
    %114 = "mhlo.convert"(%36) : (tensor<128xi64>) -> tensor<128xf64>
    %115 = "mhlo.compare"(%114, %24) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    %116 = "mhlo.broadcast_in_dim"(%115) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %117 = "mhlo.get_tuple_element"(%101) {index = 0 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x128xf32>
    %118 = mhlo.multiply %86, %32 : tensor<1x128x128xf32>
    %119 = mhlo.multiply %56, %56 : tensor<1x128x128xf32>
    %120 = mhlo.multiply %119, %34 : tensor<1x128x128xf32>
    %121 = "mhlo.exponential"(%120) : (tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    %122 = mhlo.multiply %56, %121 : tensor<1x128x128xf32>
    %123 = mhlo.multiply %122, %35 : tensor<1x128x128xf32>
    %124 = mhlo.add %118, %123 : tensor<1x128x128xf32>
    %125 = mhlo.multiply %117, %124 : tensor<1x128x128xf32>
    %126 = "mhlo.reshape"(%125) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %127 = "mhlo.dot"(%126, %arg38) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %128 = "mhlo.select"(%116, %127, %2) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %129 = "mhlo.scatter"(%26, %113, %128) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %156 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%156) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<30522x128xf32>
    %130 = "mhlo.compare"(%47, %5) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %131 = mhlo.add %47, %7 : tensor<128xi64>
    %132 = "mhlo.select"(%130, %131, %47) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %133 = "mhlo.reshape"(%132) : (tensor<128xi64>) -> tensor<128x1xi64>
    %134 = "mhlo.convert"(%47) : (tensor<128xi64>) -> tensor<128xf64>
    %135 = "mhlo.compare"(%134, %3) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    %136 = "mhlo.broadcast_in_dim"(%135) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %137 = "mhlo.select"(%136, %127, %2) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %138 = "mhlo.scatter"(%8, %133, %137) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %156 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%156) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %139 = "mhlo.compare"(%41, %5) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %140 = mhlo.add %41, %4 : tensor<128xi64>
    %141 = "mhlo.select"(%139, %140, %41) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %142 = "mhlo.reshape"(%141) : (tensor<128xi64>) -> tensor<128x1xi64>
    %143 = "mhlo.convert"(%41) : (tensor<128xi64>) -> tensor<128xf64>
    %144 = "mhlo.compare"(%143, %3) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    %145 = "mhlo.broadcast_in_dim"(%144) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %146 = "mhlo.select"(%145, %127, %2) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %147 = "mhlo.scatter"(%6, %142, %146) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %156 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%156) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %148 = call @MatmulOp0(%52, %126) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %149 = "mhlo.reduce"(%125, %0) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %156 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%156) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %150 = mhlo.multiply %95, %89 : tensor<1x128x128xf32>
    %151 = "mhlo.reduce"(%150, %0) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %156 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%156) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %152 = "mhlo.reduce"(%95, %0) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %156 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%156) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %153 = call @MatmulOp1(%105, %29) : (tensor<128x128xf32>, tensor<128x30522xf32>) -> tensor<30522x128xf32>
    %154 = "mhlo.reduce"(%30, %0) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %156 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%156) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x30522xf32>, tensor<f32>) -> tensor<30522xf32>
    %155 = "mhlo.tuple"(%109, %129, %138, %147, %148, %149, %151, %152, %153, %154) : (tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>) -> tuple<tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>
    return %155 : tuple<tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>
  }
}

// CHECK-LABEL: func private @MatmulOp0(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"}

// CHECK-LABEL: func private @MatmulOp1(%arg0: tensor<128x128xf32>, %arg1: tensor<128x30522xf32>) -> tensor<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"}

// CHECK-LABEL: func private @Unknown0

// CHECK-LABEL: func private @Unknown1

// CHECK-LABEL: func @main
