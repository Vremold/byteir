// RUN: byteir-opt %s --mhlo-arith-opt --cse --canonicalize  | FileCheck %s

module  {
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
    %56 = "mhlo.transpose"(%arg38) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %57 = "mhlo.dot"(%55, %56) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %58 = "mhlo.reshape"(%57) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %59 = mhlo.multiply %arg39, %8 : tensor<128xf32>
    %60 = "mhlo.broadcast_in_dim"(%59) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %61 = mhlo.add %58, %60 : tensor<1x128x128xf32>
    %62 = mhlo.multiply %61, %3 : tensor<1x128x128xf32>
    %63 = mhlo.multiply %61, %2 : tensor<1x128x128xf32>
    %64 = "mhlo.clamp"(%27, %63, %26) : (tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    %65 = mhlo.multiply %64, %64 : tensor<1x128x128xf32>
    %66 = mhlo.multiply %65, %18 : tensor<1x128x128xf32>
    %67 = mhlo.add %66, %25 : tensor<1x128x128xf32>
    %68 = mhlo.multiply %67, %65 : tensor<1x128x128xf32>
    %69 = mhlo.add %68, %24 : tensor<1x128x128xf32>
    %70 = mhlo.multiply %69, %65 : tensor<1x128x128xf32>
    %71 = mhlo.add %70, %23 : tensor<1x128x128xf32>
    %72 = mhlo.multiply %71, %65 : tensor<1x128x128xf32>
    %73 = mhlo.add %72, %22 : tensor<1x128x128xf32>
    %74 = mhlo.multiply %73, %65 : tensor<1x128x128xf32>
    %75 = mhlo.add %74, %21 : tensor<1x128x128xf32>
    %76 = mhlo.multiply %75, %65 : tensor<1x128x128xf32>
    %77 = mhlo.add %76, %20 : tensor<1x128x128xf32>
    %78 = mhlo.multiply %77, %65 : tensor<1x128x128xf32>
    %79 = mhlo.add %78, %19 : tensor<1x128x128xf32>
    %80 = mhlo.multiply %64, %79 : tensor<1x128x128xf32>
    %81 = mhlo.add %66, %17 : tensor<1x128x128xf32>
    %82 = mhlo.multiply %81, %65 : tensor<1x128x128xf32>
    %83 = mhlo.add %82, %16 : tensor<1x128x128xf32>
    %84 = mhlo.multiply %83, %65 : tensor<1x128x128xf32>
    %85 = mhlo.add %84, %15 : tensor<1x128x128xf32>
    %86 = mhlo.multiply %85, %65 : tensor<1x128x128xf32>
    %87 = mhlo.add %86, %14 : tensor<1x128x128xf32>
    %88 = mhlo.multiply %87, %65 : tensor<1x128x128xf32>
    %89 = mhlo.add %88, %13 : tensor<1x128x128xf32>
    %90 = mhlo.divide %80, %89 : tensor<1x128x128xf32>
    %91 = mhlo.add %90, %35 : tensor<1x128x128xf32>
    %92 = mhlo.multiply %62, %91 : tensor<1x128x128xf32>
    %93 = "mhlo.batch_norm_training"(%92, %8, %4) {epsilon = 9.99999996E-13 : f32, feature_index = 1 : i64} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %94 = "mhlo.get_tuple_element"(%93) {index = 0 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x128xf32>
    %95 = "mhlo.get_tuple_element"(%93) {index = 1 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %96 = "mhlo.get_tuple_element"(%93) {index = 2 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %97 = mhlo.add %96, %7 : tensor<128xf32>
    %98 = "mhlo.rsqrt"(%97) : (tensor<128xf32>) -> tensor<128xf32>
    %99 = "mhlo.transpose"(%arg42) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<30522x128xf32>) -> tensor<128x30522xf32>
    %100 = "mhlo.transpose"(%99) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    %101 = "mhlo.dot"(%6, %100) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x30522xf32>, tensor<30522x128xf32>) -> tensor<128x128xf32>
    %102 = "mhlo.reshape"(%101) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %103 = mhlo.multiply %arg40, %8 : tensor<128xf32>
    %104 = "mhlo.broadcast_in_dim"(%103) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %105 = mhlo.multiply %102, %104 : tensor<1x128x128xf32>
    %106 = mhlo.divide %8, %98 : tensor<128xf32>
    %107 = mhlo.multiply %106, %106 : tensor<128xf32>
    %108 = mhlo.subtract %107, %7 : tensor<128xf32>
    %109 = "mhlo.batch_norm_grad"(%92, %8, %95, %108, %105) {epsilon = 9.99999996E-13 : f32, feature_index = 1 : i64} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x128xf32>) -> tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %110 = "mhlo.broadcast_in_dim"(%arg40) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %111 = mhlo.multiply %94, %110 : tensor<1x128x128xf32>
    %112 = mhlo.multiply %111, %35 : tensor<1x128x128xf32>
    %113 = "mhlo.broadcast_in_dim"(%arg41) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128x128xf32>
    %114 = mhlo.add %113, %112 : tensor<1x128x128xf32>
    %115 = "mhlo.reshape"(%114) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %116 = "mhlo.dot"(%115, %99) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x30522xf32>) -> tensor<128x30522xf32>
    %117 = "mhlo.reshape"(%116) : (tensor<128x30522xf32>) -> tensor<1x128x30522xf32>
    %118 = mhlo.multiply %arg43, %9 : tensor<30522xf32>
    %119 = "mhlo.broadcast_in_dim"(%118) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<30522xf32>) -> tensor<1x128x30522xf32>
    %120 = mhlo.add %117, %119 : tensor<1x128x30522xf32>
    %121 = "mhlo.compare"(%37, %31) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %122 = mhlo.add %37, %11 : tensor<128xi64>
    %123 = "mhlo.select"(%121, %122, %37) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %124 = "mhlo.reshape"(%123) : (tensor<128xi64>) -> tensor<128x1xi64>
    %125 = "mhlo.convert"(%37) : (tensor<128xi64>) -> tensor<128xf64>
    %126 = "mhlo.compare"(%125, %12) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    %127 = "mhlo.broadcast_in_dim"(%126) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %128 = "mhlo.get_tuple_element"(%109) {index = 0 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x128xf32>
    %129 = mhlo.multiply %91, %3 : tensor<1x128x128xf32>
    %130 = mhlo.multiply %61, %61 : tensor<1x128x128xf32>
    %131 = mhlo.multiply %130, %1 : tensor<1x128x128xf32>
    %132 = "mhlo.exponential"(%131) : (tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    %133 = mhlo.multiply %61, %132 : tensor<1x128x128xf32>
    %134 = mhlo.multiply %133, %0 : tensor<1x128x128xf32>
    %135 = mhlo.add %129, %134 : tensor<1x128x128xf32>
    %136 = mhlo.multiply %128, %135 : tensor<1x128x128xf32>
    %137 = "mhlo.reshape"(%136) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %138 = "mhlo.transpose"(%56) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %139 = "mhlo.dot"(%137, %138) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %140 = "mhlo.select"(%127, %139, %34) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %141 = "mhlo.scatter"(%10, %124, %140) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %173 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%173) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<30522x128xf32>
    %142 = "mhlo.compare"(%49, %31) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %143 = mhlo.add %49, %29 : tensor<128xi64>
    %144 = "mhlo.select"(%142, %143, %49) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %145 = "mhlo.reshape"(%144) : (tensor<128xi64>) -> tensor<128x1xi64>
    %146 = "mhlo.convert"(%49) : (tensor<128xi64>) -> tensor<128xf64>
    %147 = "mhlo.compare"(%146, %33) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    %148 = "mhlo.broadcast_in_dim"(%147) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %149 = "mhlo.select"(%148, %139, %34) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %150 = "mhlo.scatter"(%28, %145, %149) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %173 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%173) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %151 = "mhlo.compare"(%42, %31) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %152 = mhlo.add %42, %32 : tensor<128xi64>
    %153 = "mhlo.select"(%151, %152, %42) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %154 = "mhlo.reshape"(%153) : (tensor<128xi64>) -> tensor<128x1xi64>
    %155 = "mhlo.convert"(%42) : (tensor<128xi64>) -> tensor<128xf64>
    %156 = "mhlo.compare"(%155, %33) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    %157 = "mhlo.broadcast_in_dim"(%156) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %158 = "mhlo.select"(%157, %139, %34) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %159 = "mhlo.scatter"(%30, %154, %158) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %173 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%173) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %160 = "mhlo.transpose"(%55) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %161 = "mhlo.dot"(%160, %137) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %162 = "mhlo.transpose"(%161) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %163 = "mhlo.reduce"(%136, %36) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %173 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%173) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %164 = mhlo.multiply %94, %35 : tensor<1x128x128xf32>
    %165 = mhlo.multiply %102, %164 : tensor<1x128x128xf32>
    %166 = "mhlo.reduce"(%165, %36) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %173 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%173) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %167 = "mhlo.reduce"(%102, %36) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %173 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%173) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %168 = "mhlo.transpose"(%115) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %169 = "mhlo.dot"(%168, %6) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x30522xf32>) -> tensor<128x30522xf32>
    %170 = "mhlo.transpose"(%169) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    %171 = "mhlo.reduce"(%5, %36) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %173 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%173) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x30522xf32>, tensor<f32>) -> tensor<30522xf32>
    %172 = "mhlo.tuple"(%120, %141, %150, %159, %162, %163, %166, %167, %170, %171) : (tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>) -> tuple<tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>
    return %172 : tuple<tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>
  }
}

// CHECK-LABEL: func @main
