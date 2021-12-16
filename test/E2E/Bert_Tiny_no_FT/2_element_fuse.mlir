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
    %0 = mhlo.constant dense<0.398942292> : tensor<1x128x128xf32>
    %1 = mhlo.constant dense<-5.000000e-01> : tensor<1x128x128xf32>
    %2 = mhlo.constant dense<0.707106769> : tensor<1x128x128xf32>
    %3 = mhlo.constant dense<5.000000e-01> : tensor<1x128x128xf32>
    %4 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %5 = mhlo.constant dense<1.000000e+00> : tensor<1x128x30522xf32>
    %6 = mhlo.constant dense<1.000000e+00> : tensor<128x30522xf32>
    %7 = mhlo.constant dense<9.99999996E-13> : tensor<128xf32>
    %8 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %9 = mhlo.constant dense<1.000000e+00> : tensor<30522xf32>
    %10 = mhlo.constant dense<0.000000e+00> : tensor<30522x128xf32>
    %11 = mhlo.constant dense<30522> : tensor<128xi64>
    %12 = mhlo.constant dense<0.000000e+00> : tensor<128xf64>
    %13 = mhlo.constant dense<-0.0142647391> : tensor<1x128x128xf32>
    %14 = mhlo.constant dense<-0.00737332925> : tensor<1x128x128xf32>
    %15 = mhlo.constant dense<-0.00168282702> : tensor<1x128x128xf32>
    %16 = mhlo.constant dense<-2.13374049E-4> : tensor<1x128x128xf32>
    %17 = mhlo.constant dense<-1.45660715E-5> : tensor<1x128x128xf32>
    %18 = mhlo.constant dense<0.000000e+00> : tensor<1x128x128xf32>
    %19 = mhlo.constant dense<-0.0160960332> : tensor<1x128x128xf32>
    %20 = mhlo.constant dense<-2.954600e-03> : tensor<1x128x128xf32>
    %21 = mhlo.constant dense<-7.34990637E-4> : tensor<1x128x128xf32>
    %22 = mhlo.constant dense<-5.69250624E-5> : tensor<1x128x128xf32>
    %23 = mhlo.constant dense<-2.10102394E-6> : tensor<1x128x128xf32>
    %24 = mhlo.constant dense<2.77068146E-8> : tensor<1x128x128xf32>
    %25 = mhlo.constant dense<-2.72614237E-10> : tensor<1x128x128xf32>
    %26 = mhlo.constant dense<4.000000e+00> : tensor<1x128x128xf32>
    %27 = mhlo.constant dense<-4.000000e+00> : tensor<1x128x128xf32>
    %28 = mhlo.constant dense<0.000000e+00> : tensor<512x128xf32>
    %29 = mhlo.constant dense<512> : tensor<128xi64>
    %30 = mhlo.constant dense<0.000000e+00> : tensor<2x128xf32>
    %31 = mhlo.constant dense<0> : tensor<128xi64>
    %32 = mhlo.constant dense<2> : tensor<128xi64>
    %33 = mhlo.constant dense<-1.000000e+00> : tensor<128xf64>
    %34 = mhlo.constant dense<0.000000e+00> : tensor<128x128xf32>
    %35 = mhlo.constant dense<1.000000e+00> : tensor<1x128x128xf32>
    %36 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %37 = "mhlo.reshape"(%arg46) : (tensor<1x128xi64>) -> tensor<128xi64>
    %38 = "mhlo.convert"(%37) : (tensor<128xi64>) -> tensor<128xui32>
    %39 = "mhlo.gather"(%arg0, %38) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %40 = "mhlo.reshape"(%39) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %41 = "mhlo.slice"(%arg45) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %42 = "mhlo.reshape"(%41) : (tensor<1x128xi64>) -> tensor<128xi64>
    %43 = "mhlo.convert"(%42) : (tensor<128xi64>) -> tensor<128xui32>
    %44 = "mhlo.gather"(%arg2, %43) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<2x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %45 = "mhlo.reshape"(%44) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %46 = mhlo.multiply %45, %35 : tensor<1x128x128xf32>
    %47 = mhlo.add %40, %46 : tensor<1x128x128xf32>
    %48 = "mhlo.slice"(%arg44) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %49 = "mhlo.reshape"(%48) : (tensor<1x128xi64>) -> tensor<128xi64>
    %50 = "mhlo.convert"(%49) : (tensor<128xi64>) -> tensor<128xui32>
    %51 = "mhlo.gather"(%arg1, %50) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<512x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %52 = "mhlo.reshape"(%51) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %53 = mhlo.multiply %52, %35 : tensor<1x128x128xf32>
    %54 = mhlo.add %47, %53 : tensor<1x128x128xf32>
    %55 = "mhlo.reshape"(%54) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %56 = "mhlo.dot_general"(%55, %arg38) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %57 = "mhlo.reshape"(%56) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %58 = mhlo.multiply %arg39, %8 : tensor<128xf32>
    %59 = "mhlo.broadcast_in_dim"(%58) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %60 = mhlo.add %57, %59 : tensor<1x128x128xf32>
    %61 = mhlo.multiply %60, %3 : tensor<1x128x128xf32>
    %62 = mhlo.multiply %60, %2 : tensor<1x128x128xf32>
    %63 = "mhlo.clamp"(%27, %62, %26) : (tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    %64 = mhlo.multiply %63, %63 : tensor<1x128x128xf32>
    %65 = mhlo.multiply %64, %18 : tensor<1x128x128xf32>
    %66 = mhlo.add %65, %25 : tensor<1x128x128xf32>
    %67 = mhlo.multiply %66, %64 : tensor<1x128x128xf32>
    %68 = mhlo.add %67, %24 : tensor<1x128x128xf32>
    %69 = mhlo.multiply %68, %64 : tensor<1x128x128xf32>
    %70 = mhlo.add %69, %23 : tensor<1x128x128xf32>
    %71 = mhlo.multiply %70, %64 : tensor<1x128x128xf32>
    %72 = mhlo.add %71, %22 : tensor<1x128x128xf32>
    %73 = mhlo.multiply %72, %64 : tensor<1x128x128xf32>
    %74 = mhlo.add %73, %21 : tensor<1x128x128xf32>
    %75 = mhlo.multiply %74, %64 : tensor<1x128x128xf32>
    %76 = mhlo.add %75, %20 : tensor<1x128x128xf32>
    %77 = mhlo.multiply %76, %64 : tensor<1x128x128xf32>
    %78 = mhlo.add %77, %19 : tensor<1x128x128xf32>
    %79 = mhlo.multiply %63, %78 : tensor<1x128x128xf32>
    %80 = mhlo.add %65, %17 : tensor<1x128x128xf32>
    %81 = mhlo.multiply %80, %64 : tensor<1x128x128xf32>
    %82 = mhlo.add %81, %16 : tensor<1x128x128xf32>
    %83 = mhlo.multiply %82, %64 : tensor<1x128x128xf32>
    %84 = mhlo.add %83, %15 : tensor<1x128x128xf32>
    %85 = mhlo.multiply %84, %64 : tensor<1x128x128xf32>
    %86 = mhlo.add %85, %14 : tensor<1x128x128xf32>
    %87 = mhlo.multiply %86, %64 : tensor<1x128x128xf32>
    %88 = mhlo.add %87, %13 : tensor<1x128x128xf32>
    %89 = mhlo.divide %79, %88 : tensor<1x128x128xf32>
    %90 = mhlo.add %89, %35 : tensor<1x128x128xf32>
    %91 = mhlo.multiply %61, %90 : tensor<1x128x128xf32>
    %92 = "mhlo.batch_norm_training"(%91, %8, %4) {epsilon = 9.99999996E-13 : f32, feature_index = 1 : i64} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %93 = "mhlo.get_tuple_element"(%92) {index = 0 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x128xf32>
    %94 = "mhlo.get_tuple_element"(%92) {index = 1 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %95 = "mhlo.get_tuple_element"(%92) {index = 2 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %96 = mhlo.add %95, %7 : tensor<128xf32>
    %97 = "mhlo.rsqrt"(%96) : (tensor<128xf32>) -> tensor<128xf32>
    %98 = "mhlo.dot"(%6, %arg42) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x30522xf32>, tensor<30522x128xf32>) -> tensor<128x128xf32>
    %99 = "mhlo.reshape"(%98) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %100 = mhlo.multiply %arg40, %8 : tensor<128xf32>
    %101 = "mhlo.broadcast_in_dim"(%100) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %102 = mhlo.multiply %99, %101 : tensor<1x128x128xf32>
    %103 = mhlo.divide %8, %97 : tensor<128xf32>
    %104 = mhlo.multiply %103, %103 : tensor<128xf32>
    %105 = mhlo.subtract %104, %7 : tensor<128xf32>
    %106 = "mhlo.batch_norm_grad"(%91, %8, %94, %105, %102) {epsilon = 9.99999996E-13 : f32, feature_index = 1 : i64} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x128xf32>) -> tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %107 = "mhlo.broadcast_in_dim"(%arg40) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %108 = mhlo.multiply %93, %107 : tensor<1x128x128xf32>
    %109 = mhlo.multiply %108, %35 : tensor<1x128x128xf32>
    %110 = "mhlo.broadcast_in_dim"(%arg41) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %111 = mhlo.add %110, %109 : tensor<1x128x128xf32>
    %112 = "mhlo.reshape"(%111) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %113 = "mhlo.dot_general"(%112, %arg42) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<30522x128xf32>) -> tensor<128x30522xf32>
    %114 = "mhlo.reshape"(%113) : (tensor<128x30522xf32>) -> tensor<1x128x30522xf32>
    %115 = mhlo.multiply %arg43, %9 : tensor<30522xf32>
    %116 = "mhlo.broadcast_in_dim"(%115) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<30522xf32>) -> tensor<1x128x30522xf32>
    %117 = mhlo.add %114, %116 : tensor<1x128x30522xf32>
    %118 = "mhlo.compare"(%37, %31) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %119 = mhlo.add %37, %11 : tensor<128xi64>
    %120 = "mhlo.select"(%118, %119, %37) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %121 = "mhlo.reshape"(%120) : (tensor<128xi64>) -> tensor<128x1xi64>
    %122 = "mhlo.convert"(%37) : (tensor<128xi64>) -> tensor<128xf64>
    %123 = "mhlo.compare"(%122, %12) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    %124 = "mhlo.broadcast_in_dim"(%123) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %125 = "mhlo.get_tuple_element"(%106) {index = 0 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x128xf32>
    %126 = mhlo.multiply %90, %3 : tensor<1x128x128xf32>
    %127 = mhlo.multiply %60, %60 : tensor<1x128x128xf32>
    %128 = mhlo.multiply %127, %1 : tensor<1x128x128xf32>
    %129 = "mhlo.exponential"(%128) : (tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    %130 = mhlo.multiply %60, %129 : tensor<1x128x128xf32>
    %131 = mhlo.multiply %130, %0 : tensor<1x128x128xf32>
    %132 = mhlo.add %126, %131 : tensor<1x128x128xf32>
    %133 = mhlo.multiply %125, %132 : tensor<1x128x128xf32>
    %134 = "mhlo.reshape"(%133) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %135 = "mhlo.dot"(%134, %arg38) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %136 = "mhlo.select"(%124, %135, %34) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %137 = "mhlo.scatter"(%10, %121, %136) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %165 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%165) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<30522x128xf32>
    %138 = "mhlo.compare"(%49, %31) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %139 = mhlo.add %49, %29 : tensor<128xi64>
    %140 = "mhlo.select"(%138, %139, %49) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %141 = "mhlo.reshape"(%140) : (tensor<128xi64>) -> tensor<128x1xi64>
    %142 = "mhlo.convert"(%49) : (tensor<128xi64>) -> tensor<128xf64>
    %143 = "mhlo.compare"(%142, %33) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    %144 = "mhlo.broadcast_in_dim"(%143) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %145 = "mhlo.select"(%144, %135, %34) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %146 = "mhlo.scatter"(%28, %141, %145) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %165 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%165) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %147 = "mhlo.compare"(%42, %31) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %148 = mhlo.add %42, %32 : tensor<128xi64>
    %149 = "mhlo.select"(%147, %148, %42) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %150 = "mhlo.reshape"(%149) : (tensor<128xi64>) -> tensor<128x1xi64>
    %151 = "mhlo.convert"(%42) : (tensor<128xi64>) -> tensor<128xf64>
    %152 = "mhlo.compare"(%151, %33) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    %153 = "mhlo.broadcast_in_dim"(%152) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %154 = "mhlo.select"(%153, %135, %34) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %155 = "mhlo.scatter"(%30, %150, %154) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %165 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%165) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %156 = call @MatmulOp0(%55, %134) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %157 = "mhlo.reduce"(%133, %36) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %165 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%165) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %158 = mhlo.multiply %93, %35 : tensor<1x128x128xf32>
    %159 = mhlo.multiply %99, %158 : tensor<1x128x128xf32>
    %160 = "mhlo.reduce"(%159, %36) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %165 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%165) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %161 = "mhlo.reduce"(%99, %36) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %165 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%165) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %162 = call @MatmulOp1(%112, %6) : (tensor<128x128xf32>, tensor<128x30522xf32>) -> tensor<30522x128xf32>
    %163 = "mhlo.reduce"(%5, %36) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %165 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%165) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x30522xf32>, tensor<f32>) -> tensor<30522xf32>
    %164 = "mhlo.tuple"(%117, %137, %146, %155, %156, %157, %160, %161, %162, %163) : (tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>) -> tuple<tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>
    return %164 : tuple<tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>
  }
}

// CHECK-LABEL: func private @MatmulOp0(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"}

// CHECK-LABEL: func private @MatmulOp1(%arg0: tensor<128x128xf32>, %arg1: tensor<128x30522xf32>) -> tensor<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"}

// CHECK-LABEL: func private @Unknown0

// CHECK-LABEL: func private @Unknown1

// CHECK-LABEL: func @main
