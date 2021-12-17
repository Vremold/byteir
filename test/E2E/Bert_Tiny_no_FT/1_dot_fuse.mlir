// RUN: byteir-opt %s --hlo-fold --hlo-transpose-dot-to-dot-general --fuse-dot-transpose --fusion-outlining | FileCheck %s

module  {
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
    %53 = "mhlo.transpose"(%arg38) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %54 = "mhlo.dot"(%52, %53) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %55 = "mhlo.reshape"(%54) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %56 = "mhlo.broadcast_in_dim"(%arg39) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %57 = mhlo.add %55, %56 : tensor<1x128x128xf32>
    %58 = mhlo.multiply %57, %32 : tensor<1x128x128xf32>
    %59 = mhlo.multiply %57, %33 : tensor<1x128x128xf32>
    %60 = "mhlo.clamp"(%9, %59, %10) : (tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    %61 = mhlo.multiply %60, %60 : tensor<1x128x128xf32>
    %62 = mhlo.multiply %61, %18 : tensor<1x128x128xf32>
    %63 = mhlo.add %62, %11 : tensor<1x128x128xf32>
    %64 = mhlo.multiply %63, %61 : tensor<1x128x128xf32>
    %65 = mhlo.add %64, %12 : tensor<1x128x128xf32>
    %66 = mhlo.multiply %65, %61 : tensor<1x128x128xf32>
    %67 = mhlo.add %66, %13 : tensor<1x128x128xf32>
    %68 = mhlo.multiply %67, %61 : tensor<1x128x128xf32>
    %69 = mhlo.add %68, %14 : tensor<1x128x128xf32>
    %70 = mhlo.multiply %69, %61 : tensor<1x128x128xf32>
    %71 = mhlo.add %70, %15 : tensor<1x128x128xf32>
    %72 = mhlo.multiply %71, %61 : tensor<1x128x128xf32>
    %73 = mhlo.add %72, %16 : tensor<1x128x128xf32>
    %74 = mhlo.multiply %73, %61 : tensor<1x128x128xf32>
    %75 = mhlo.add %74, %17 : tensor<1x128x128xf32>
    %76 = mhlo.multiply %60, %75 : tensor<1x128x128xf32>
    %77 = mhlo.add %62, %19 : tensor<1x128x128xf32>
    %78 = mhlo.multiply %77, %61 : tensor<1x128x128xf32>
    %79 = mhlo.add %78, %20 : tensor<1x128x128xf32>
    %80 = mhlo.multiply %79, %61 : tensor<1x128x128xf32>
    %81 = mhlo.add %80, %21 : tensor<1x128x128xf32>
    %82 = mhlo.multiply %81, %61 : tensor<1x128x128xf32>
    %83 = mhlo.add %82, %22 : tensor<1x128x128xf32>
    %84 = mhlo.multiply %83, %61 : tensor<1x128x128xf32>
    %85 = mhlo.add %84, %23 : tensor<1x128x128xf32>
    %86 = mhlo.divide %76, %85 : tensor<1x128x128xf32>
    %87 = mhlo.add %86, %1 : tensor<1x128x128xf32>
    %88 = mhlo.multiply %58, %87 : tensor<1x128x128xf32>
    %89 = "mhlo.batch_norm_training"(%88, %27, %31) {epsilon = 9.99999996E-13 : f32, feature_index = 1 : i64} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %90 = "mhlo.get_tuple_element"(%89) {index = 0 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x128xf32>
    %91 = "mhlo.get_tuple_element"(%89) {index = 1 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %92 = "mhlo.get_tuple_element"(%89) {index = 2 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %93 = mhlo.add %92, %28 : tensor<128xf32>
    %94 = "mhlo.rsqrt"(%93) : (tensor<128xf32>) -> tensor<128xf32>
    %95 = "mhlo.transpose"(%arg42) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<30522x128xf32>) -> tensor<128x30522xf32>
    %96 = "mhlo.transpose"(%95) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    %97 = "mhlo.dot"(%29, %96) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x30522xf32>, tensor<30522x128xf32>) -> tensor<128x128xf32>
    %98 = "mhlo.reshape"(%97) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %99 = "mhlo.broadcast_in_dim"(%arg40) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %100 = mhlo.multiply %98, %99 : tensor<1x128x128xf32>
    %101 = mhlo.divide %27, %94 : tensor<128xf32>
    %102 = mhlo.multiply %101, %101 : tensor<128xf32>
    %103 = mhlo.subtract %102, %28 : tensor<128xf32>
    %104 = "mhlo.batch_norm_grad"(%88, %27, %91, %103, %100) {epsilon = 9.99999996E-13 : f32, feature_index = 1 : i64} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x128xf32>) -> tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %105 = mhlo.multiply %90, %99 : tensor<1x128x128xf32>
    %106 = "mhlo.broadcast_in_dim"(%arg41) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %107 = mhlo.add %106, %105 : tensor<1x128x128xf32>
    %108 = "mhlo.reshape"(%107) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %109 = "mhlo.dot"(%108, %95) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x30522xf32>) -> tensor<128x30522xf32>
    %110 = "mhlo.reshape"(%109) : (tensor<128x30522xf32>) -> tensor<1x128x30522xf32>
    %111 = "mhlo.broadcast_in_dim"(%arg43) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<30522xf32>) -> tensor<1x128x30522xf32>
    %112 = mhlo.add %110, %111 : tensor<1x128x30522xf32>
    %113 = "mhlo.compare"(%36, %5) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %114 = mhlo.add %36, %25 : tensor<128xi64>
    %115 = "mhlo.select"(%113, %114, %36) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %116 = "mhlo.reshape"(%115) : (tensor<128xi64>) -> tensor<128x1xi64>
    %117 = "mhlo.convert"(%36) : (tensor<128xi64>) -> tensor<128xf64>
    %118 = "mhlo.compare"(%117, %24) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    %119 = "mhlo.broadcast_in_dim"(%118) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %120 = "mhlo.get_tuple_element"(%104) {index = 0 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x128xf32>
    %121 = mhlo.multiply %87, %32 : tensor<1x128x128xf32>
    %122 = mhlo.multiply %57, %57 : tensor<1x128x128xf32>
    %123 = mhlo.multiply %122, %34 : tensor<1x128x128xf32>
    %124 = "mhlo.exponential"(%123) : (tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    %125 = mhlo.multiply %57, %124 : tensor<1x128x128xf32>
    %126 = mhlo.multiply %125, %35 : tensor<1x128x128xf32>
    %127 = mhlo.add %121, %126 : tensor<1x128x128xf32>
    %128 = mhlo.multiply %120, %127 : tensor<1x128x128xf32>
    %129 = "mhlo.reshape"(%128) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %130 = "mhlo.transpose"(%53) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %131 = "mhlo.dot"(%129, %130) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %132 = "mhlo.select"(%119, %131, %2) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %133 = "mhlo.scatter"(%26, %116, %132) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %164 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%164) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<30522x128xf32>
    %134 = "mhlo.compare"(%47, %5) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %135 = mhlo.add %47, %7 : tensor<128xi64>
    %136 = "mhlo.select"(%134, %135, %47) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %137 = "mhlo.reshape"(%136) : (tensor<128xi64>) -> tensor<128x1xi64>
    %138 = "mhlo.convert"(%47) : (tensor<128xi64>) -> tensor<128xf64>
    %139 = "mhlo.compare"(%138, %3) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    %140 = "mhlo.broadcast_in_dim"(%139) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %141 = "mhlo.select"(%140, %131, %2) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %142 = "mhlo.scatter"(%8, %137, %141) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %164 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%164) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %143 = "mhlo.compare"(%41, %5) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %144 = mhlo.add %41, %4 : tensor<128xi64>
    %145 = "mhlo.select"(%143, %144, %41) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %146 = "mhlo.reshape"(%145) : (tensor<128xi64>) -> tensor<128x1xi64>
    %147 = "mhlo.convert"(%41) : (tensor<128xi64>) -> tensor<128xf64>
    %148 = "mhlo.compare"(%147, %3) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    %149 = "mhlo.broadcast_in_dim"(%148) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %150 = "mhlo.select"(%149, %131, %2) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %151 = "mhlo.scatter"(%6, %146, %150) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %164 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%164) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %152 = "mhlo.transpose"(%52) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %153 = "mhlo.dot"(%152, %129) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %154 = "mhlo.transpose"(%153) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %155 = "mhlo.reduce"(%128, %0) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %164 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%164) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %156 = mhlo.multiply %98, %90 : tensor<1x128x128xf32>
    %157 = "mhlo.reduce"(%156, %0) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %164 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%164) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %158 = "mhlo.reduce"(%98, %0) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %164 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%164) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %159 = "mhlo.transpose"(%108) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %160 = "mhlo.dot"(%159, %29) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x30522xf32>) -> tensor<128x30522xf32>
    %161 = "mhlo.transpose"(%160) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    %162 = "mhlo.reduce"(%30, %0) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %164 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%164) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x30522xf32>, tensor<f32>) -> tensor<30522xf32>
    %163 = "mhlo.tuple"(%112, %133, %142, %151, %154, %155, %157, %158, %161, %162) : (tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>) -> tuple<tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>
    return %163 : tuple<tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>
  }
}

// CHECK-LABEL: func private @MatmulOp0(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"}

// CHECK-LABEL: func private @MatmulOp1(%arg0: tensor<128x128xf32>, %arg1: tensor<128x30522xf32>) -> tensor<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"}

// CHECK-LABEL: func @main
// CHECK: mhlo.dot_general