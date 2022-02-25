// RUN: byteir-opt %s -gpu-opt | FileCheck %s

// CHECK-LABEL: func @main
#map0 = affine_map<(d0) -> (d0 mod 128)>
#map1 = affine_map<(d0) -> (d0 floordiv 128)>
#map2 = affine_map<(d0) -> ((d0 floordiv 128) mod 128)>
#map3 = affine_map<(d0) -> ((d0 floordiv 128) floordiv 128)>
#map4 = affine_map<(d0) -> (d0 mod 30522)>
#map5 = affine_map<(d0) -> ((d0 floordiv 30522) mod 128)>
#map6 = affine_map<(d0) -> ((d0 floordiv 30522) floordiv 128)>
#map7 = affine_map<(d0) -> (d0 floordiv 30522)>
module {
  func private @MatmulOp0(%arg0: memref<256x128xf32>, %arg1: memref<256x30522xf32>) -> memref<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = memref.alloc() : memref<128x30522xf32>
    %1 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x128xf32>, memref<256x30522xf32>, memref<128x30522xf32>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<128x30522xf32>, memref<30522x128xf32>) -> ()
    return %1 : memref<30522x128xf32>
  }
  func private @Unknown0(%arg0: memref<2x128xi64>) -> (memref<256xi64>, memref<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %c-100_i64 = arith.constant -100 : i64
    %0 = memref.alloc() : memref<256xi1>
    %1 = memref.collapse_shape %arg0 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    affine.for %arg1 = 0 to 256 {
      %2 = affine.apply #map0(%arg1)
      %3 = affine.apply #map1(%arg1)
      %4 = affine.load %arg0[%3, %2] : memref<2x128xi64>
      %5 = arith.cmpi ne, %4, %c-100_i64 : i64
      affine.store %5, %0[%arg1] : memref<256xi1>
    }
    return %1, %0 : memref<256xi64>, memref<256xi1>
  }
  func private @Unknown1(%arg0: memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = memref.alloc() : memref<256xi1>
    %1 = memref.alloc() : memref<256xi64>
    %2 = memref.expand_shape %1 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    %3 = memref.alloc() : memref<256xui32>
    %c30522_i64 = arith.constant 30522 : i64
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg1 = 0 to 256 {
      %4 = affine.apply #map0(%arg1)
      %5 = affine.apply #map1(%arg1)
      %6 = affine.load %arg0[%5, %4] : memref<2x128xi64>
      %7 = arith.trunci %6 : i64 to i32
      %8 = builtin.unrealized_conversion_cast %7 : i32 to ui32
      affine.store %8, %3[%arg1] : memref<256xui32>
      %9 = arith.addi %6, %c30522_i64 : i64
      %10 = arith.cmpi slt, %6, %c0_i64 : i64
      %11 = select %10, %9, %6 : i64
      affine.store %11, %1[%arg1] : memref<256xi64>
      %12 = arith.sitofp %6 : i64 to f64
      %13 = arith.cmpf une, %12, %cst : f64
      affine.store %13, %0[%arg1] : memref<256xi1>
    }
    return %3, %2, %0 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func private @Unknown2(%arg0: memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant -1.000000e+00 : f64
    %0 = memref.alloc() : memref<2x128xi1>
    %1 = memref.collapse_shape %0 [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    %2 = memref.alloc() : memref<2x128xi64>
    %3 = memref.collapse_shape %2 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %4 = memref.expand_shape %3 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    %5 = memref.alloc() : memref<2x128xui32>
    %6 = memref.collapse_shape %5 [[0, 1]] : memref<2x128xui32> into memref<256xui32>
    %c2_i64 = arith.constant 2 : i64
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg1 = 0 to 256 {
      %7 = affine.apply #map0(%arg1)
      %8 = affine.apply #map1(%arg1)
      %9 = affine.load %arg0[%7] : memref<128xi64>
      %10 = arith.trunci %9 : i64 to i32
      %11 = builtin.unrealized_conversion_cast %10 : i32 to ui32
      affine.store %11, %5[%8, %7] : memref<2x128xui32>
      %12 = arith.addi %9, %c2_i64 : i64
      %13 = arith.cmpi slt, %9, %c0_i64 : i64
      %14 = select %13, %12, %9 : i64
      affine.store %14, %2[%8, %7] : memref<2x128xi64>
      %15 = arith.sitofp %9 : i64 to f64
      %16 = arith.cmpf une, %15, %cst : f64
      affine.store %16, %0[%8, %7] : memref<2x128xi1>
    }
    return %6, %4, %1 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func private @Unknown3(%arg0: memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant -1.000000e+00 : f64
    %0 = memref.alloc() : memref<128xi1>
    %1 = memref.alloc() : memref<128xi64>
    %2 = memref.expand_shape %1 [[0, 1]] : memref<128xi64> into memref<128x1xi64>
    %3 = memref.alloc() : memref<128xui32>
    %4 = memref.collapse_shape %arg0 [[0, 1]] : memref<1x128xi64> into memref<128xi64>
    %c512_i64 = arith.constant 512 : i64
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg1 = 0 to 128 {
      %5 = affine.load %4[%arg1] : memref<128xi64>
      %6 = arith.trunci %5 : i64 to i32
      %7 = builtin.unrealized_conversion_cast %6 : i32 to ui32
      affine.store %7, %3[%arg1] : memref<128xui32>
      %8 = arith.addi %5, %c512_i64 : i64
      %9 = arith.cmpi slt, %5, %c0_i64 : i64
      %10 = select %9, %8, %5 : i64
      affine.store %10, %1[%arg1] : memref<128xi64>
      %11 = arith.sitofp %5 : i64 to f64
      %12 = arith.cmpf une, %11, %cst : f64
      affine.store %12, %0[%arg1] : memref<128xi1>
    }
    return %3, %2, %0 : memref<128xui32>, memref<128x1xi64>, memref<128xi1>
  }
  func private @Unknown4(%arg0: memref<256x128xf32>, %arg1: memref<256x128xf32>, %arg2: memref<128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<2x128x128xf32>
    affine.for %arg3 = 0 to 32768 {
      %1 = affine.apply #map0(%arg3)
      %2 = affine.apply #map2(%arg3)
      %3 = affine.apply #map3(%arg3)
      %4 = affine.apply #map1(%arg3)
      %5 = affine.load %arg0[%4, %1] : memref<256x128xf32>
      %6 = affine.load %arg1[%4, %1] : memref<256x128xf32>
      %7 = affine.load %arg2[%2, %1] : memref<128x128xf32>
      %8 = arith.addf %5, %6 : f32
      %9 = arith.addf %8, %7 : f32
      affine.store %9, %0[%3, %2, %1] : memref<2x128x128xf32>
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown5(%arg0: memref<256x30522xf32>, %arg1: memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = memref.expand_shape %arg0 [[0, 1], [2]] : memref<256x30522xf32> into memref<2x128x30522xf32>
    %1 = memref.alloc() : memref<256x30522xf32>
    %2 = memref.alloc() : memref<2x128x30522xf32>
    affine.for %arg2 = 0 to 7813632 {
      %3 = affine.apply #map4(%arg2)
      %4 = affine.apply #map5(%arg2)
      %5 = affine.apply #map6(%arg2)
      %6 = affine.apply #map7(%arg2)
      %7 = affine.load %arg0[%6, %3] : memref<256x30522xf32>
      %8 = affine.load %arg1[%3] : memref<30522xf32>
      %9 = arith.addf %7, %8 : f32
      affine.store %9, %2[%5, %4, %3] : memref<2x128x30522xf32>
      %10 = affine.load %0[%5, %4, %3] : memref<2x128x30522xf32>
      %11 = arith.addf %10, %8 : f32
      affine.store %11, %1[%6, %3] : memref<256x30522xf32>
    }
    return %2, %1 : memref<2x128x30522xf32>, memref<256x30522xf32>
  }
  func private @Unknown6(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x30522xf32>
    %1 = memref.alloc() : memref<256x30522xf32>
    affine.for %arg2 = 0 to 7813632 {
      %2 = affine.apply #map4(%arg2)
      %3 = affine.apply #map7(%arg2)
      %4 = affine.load %arg1[%3, %2] : memref<256x30522xf32>
      %5 = affine.load %arg0[%3] : memref<256xf32>
      %6 = arith.subf %4, %5 : f32
      affine.store %6, %0[%3, %2] : memref<256x30522xf32>
      %7 = arith.subf %4, %5 : f32
      %8 = math.exp %7 : f32
      affine.store %8, %1[%3, %2] : memref<256x30522xf32>
    }
    return %0, %1 : memref<256x30522xf32>, memref<256x30522xf32>
  }
  func private @Unknown7(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256xf32>
    affine.for %arg1 = 0 to 256 {
      %1 = affine.load %arg0[%arg1] : memref<256xf32>
      %2 = math.log %1 : f32
      affine.store %2, %0[%arg1] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @Unknown8(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256xi64>, %arg3: memref<256xi1>) -> (memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = memref.alloc() : memref<256x30522xf32>
    %1 = memref.alloc() : memref<256x30522xf32>
    %2 = memref.alloc() : memref<256x30522xf32>
    %3 = memref.alloc() : memref<256x30522xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    affine.for %arg4 = 0 to 7813632 {
      %4 = affine.apply #map4(%arg4)
      %5 = affine.apply #map7(%arg4)
      %6 = affine.load %arg3[%5] : memref<256xi1>
      %7 = affine.load %arg2[%5] : memref<256xi64>
      %8 = arith.index_cast %4 : index to i64
      %9 = arith.cmpi eq, %7, %8 : i64
      %10 = select %9, %cst, %cst_0 : f32
      %11 = select %6, %cst, %cst_0 : f32
      %12 = arith.mulf %11, %10 : f32
      affine.store %12, %3[%5, %4] : memref<256x30522xf32>
      %13 = affine.load %arg1[%5, %4] : memref<256x30522xf32>
      %14 = affine.load %arg0[%5] : memref<256xf32>
      %15 = arith.cmpi eq, %7, %8 : i64
      %16 = select %15, %cst, %cst_0 : f32
      %17 = select %6, %cst, %cst_0 : f32
      %18 = arith.mulf %17, %16 : f32
      %19 = arith.subf %13, %14 : f32
      %20 = arith.negf %16 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.cmpf une, %16, %cst : f32
      %23 = select %22, %cst_0, %21 : f32
      %24 = arith.mulf %23, %18 : f32
      affine.store %24, %2[%5, %4] : memref<256x30522xf32>
      %25 = arith.cmpi eq, %7, %8 : i64
      %26 = select %25, %cst, %cst_0 : f32
      %27 = select %6, %cst, %cst_0 : f32
      %28 = arith.mulf %27, %26 : f32
      %29 = arith.negf %26 : f32
      %30 = arith.mulf %29, %28 : f32
      affine.store %30, %1[%5, %4] : memref<256x30522xf32>
      %31 = arith.subf %13, %14 : f32
      %32 = math.exp %31 : f32
      affine.store %32, %0[%5, %4] : memref<256x30522xf32>
    }
    return %3, %2, %1, %0 : memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
  }
  func private @Unknown9(%arg0: memref<f32>) -> memref<f32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = memref.alloc() : memref<f32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    affine.for %arg1 = 0 to 1 {
      %1 = affine.load %arg0[] : memref<f32>
      %2 = arith.cmpf une, %1, %cst_0 : f32
      %3 = select %2, %1, %cst : f32
      affine.store %3, %0[] : memref<f32>
    }
    return %0 : memref<f32>
  }
  func private @Unknown10(%arg0: memref<f32>, %arg1: memref<256x30522xf32>) -> memref<256x30522xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x30522xf32>
    affine.for %arg2 = 0 to 7813632 {
      %1 = affine.apply #map4(%arg2)
      %2 = affine.apply #map7(%arg2)
      %3 = affine.load %arg1[%2, %1] : memref<256x30522xf32>
      %4 = affine.load %arg0[] : memref<f32>
      %5 = arith.divf %3, %4 : f32
      affine.store %5, %0[%2, %1] : memref<256x30522xf32>
    }
    return %0 : memref<256x30522xf32>
  }
  func private @Unknown11(%arg0: memref<f32>, %arg1: memref<f32>) -> memref<f32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<f32>
    affine.for %arg2 = 0 to 1 {
      %1 = affine.load %arg0[] : memref<f32>
      %2 = affine.load %arg1[] : memref<f32>
      %3 = arith.divf %1, %2 : f32
      affine.store %3, %0[] : memref<f32>
    }
    return %0 : memref<f32>
  }
  func private @Unknown12(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<2x128x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x30522xf32>
    %1 = memref.alloc() : memref<2x128x30522xf32>
    %2 = memref.expand_shape %arg0 [[0, 1]] : memref<256xf32> into memref<2x128xf32>
    affine.for %arg3 = 0 to 7813632 {
      %3 = affine.apply #map4(%arg3)
      %4 = affine.apply #map7(%arg3)
      %5 = affine.load %arg2[%4, %3] : memref<256x30522xf32>
      %6 = affine.load %arg1[%4, %3] : memref<256x30522xf32>
      %7 = affine.load %arg0[%4] : memref<256xf32>
      %8 = arith.mulf %6, %7 : f32
      %9 = arith.subf %5, %8 : f32
      affine.store %9, %0[%4, %3] : memref<256x30522xf32>
      %10 = affine.apply #map5(%arg3)
      %11 = affine.apply #map6(%arg3)
      %12 = affine.load %2[%11, %10] : memref<2x128xf32>
      %13 = arith.mulf %6, %12 : f32
      %14 = arith.subf %5, %13 : f32
      affine.store %14, %1[%11, %10, %3] : memref<2x128x30522xf32>
    }
    return %0, %1 : memref<256x30522xf32>, memref<2x128x30522xf32>
  }
  func private @Unknown13(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<2x128x128xf32>
    affine.for %arg2 = 0 to 32768 {
      %1 = affine.apply #map0(%arg2)
      %2 = affine.apply #map2(%arg2)
      %3 = affine.apply #map3(%arg2)
      %4 = affine.load %arg0[%3, %2, %1] : memref<2x128x128xf32>
      %5 = affine.load %arg1[%3, %2, %1] : memref<2x128x128xf32>
      %6 = arith.addf %4, %5 : f32
      affine.store %6, %0[%3, %2, %1] : memref<2x128x128xf32>
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown14(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<2x128x128xf32>
    affine.for %arg4 = 0 to 32768 {
      %1 = affine.apply #map0(%arg4)
      %2 = affine.apply #map2(%arg4)
      %3 = affine.apply #map3(%arg4)
      %4 = affine.load %arg0[%3, %2, %1] : memref<2x128x128xf32>
      %5 = affine.load %arg1[%3, %2, %1] : memref<2x128x128xf32>
      %6 = affine.load %arg2[%3, %2, %1] : memref<2x128x128xf32>
      %7 = affine.load %arg3[%3, %2, %1] : memref<2x128x128xf32>
      %8 = arith.addf %4, %5 : f32
      %9 = arith.addf %8, %6 : f32
      %10 = arith.addf %9, %7 : f32
      affine.store %10, %0[%3, %2, %1] : memref<2x128x128xf32>
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown15(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<2x128x128xf32>
    affine.for %arg2 = 0 to 32768 {
      %1 = affine.apply #map0(%arg2)
      %2 = affine.apply #map2(%arg2)
      %3 = affine.apply #map3(%arg2)
      %4 = affine.load %arg0[%3, %2, %1] : memref<2x128x128xf32>
      %5 = affine.load %arg1[%3, %2, %1] : memref<2x128x128xf32>
      %6 = arith.addf %4, %5 : f32
      affine.store %6, %0[%3, %2, %1] : memref<2x128x128xf32>
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown16(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<2x128x128xf32>
    affine.for %arg4 = 0 to 32768 {
      %1 = affine.apply #map0(%arg4)
      %2 = affine.apply #map2(%arg4)
      %3 = affine.apply #map3(%arg4)
      %4 = affine.load %arg0[%3, %2, %1] : memref<2x128x128xf32>
      %5 = affine.load %arg1[%3, %2, %1] : memref<2x128x128xf32>
      %6 = affine.load %arg2[%3, %2, %1] : memref<2x128x128xf32>
      %7 = affine.load %arg3[%3, %2, %1] : memref<2x128x128xf32>
      %8 = arith.addf %4, %5 : f32
      %9 = arith.addf %8, %6 : f32
      %10 = arith.addf %9, %7 : f32
      affine.store %10, %0[%3, %2, %1] : memref<2x128x128xf32>
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown17(%arg0: memref<256xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.alloc() : memref<256x128xf32>
    %1 = memref.expand_shape %arg2 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    %2 = memref.alloc() : memref<256x128xf32>
    %3 = memref.expand_shape %arg0 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    affine.for %arg3 = 0 to 32768 {
      %4 = affine.apply #map0(%arg3)
      %5 = affine.apply #map2(%arg3)
      %6 = affine.apply #map3(%arg3)
      %7 = affine.load %3[%6, %5] : memref<2x128xi1>
      %8 = affine.load %arg1[%6, %5, %4] : memref<2x128x128xf32>
      %9 = affine.apply #map1(%arg3)
      %10 = select %7, %8, %cst : f32
      affine.store %10, %2[%9, %4] : memref<256x128xf32>
      %11 = affine.load %1[%6, %5] : memref<2x128xi1>
      %12 = select %11, %8, %cst : f32
      affine.store %12, %0[%9, %4] : memref<256x128xf32>
    }
    return %2, %0 : memref<256x128xf32>, memref<256x128xf32>
  }
  func private @Unknown18(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>) -> memref<128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.alloc() : memref<128x128xf32>
    affine.for %arg2 = 0 to 16384 {
      %1 = affine.apply #map0(%arg2)
      %2 = affine.apply #map1(%arg2)
      %3 = affine.load %arg0[%2] : memref<128xi1>
      %4 = affine.load %arg1[%2, %1] : memref<128x128xf32>
      %5 = select %3, %4, %cst : f32
      affine.store %5, %0[%2, %1] : memref<128x128xf32>
    }
    return %0 : memref<128x128xf32>
  }
  func @main(%arg0: memref<2x128xi64>, %arg1: memref<2x128xi64>, %arg2: memref<1x512xi64>, %arg3: memref<1x512xi64>, %arg4: memref<30522x128xf32>, %arg5: memref<2x128xf32>, %arg6: memref<512x128xf32>, %arg7: memref<128xf32>, %arg8: memref<128xf32>, %arg9: memref<128x128xf32>, %arg10: memref<128xf32>, %arg11: memref<128x128xf32>, %arg12: memref<128xf32>, %arg13: memref<128x128xf32>, %arg14: memref<128xf32>, %arg15: memref<128x128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<512x128xf32>, %arg20: memref<512xf32>, %arg21: memref<128x512xf32>, %arg22: memref<128xf32>, %arg23: memref<128xf32>, %arg24: memref<128xf32>, %arg25: memref<128x128xf32>, %arg26: memref<128xf32>, %arg27: memref<128x128xf32>, %arg28: memref<128xf32>, %arg29: memref<128x128xf32>, %arg30: memref<128xf32>, %arg31: memref<128x128xf32>, %arg32: memref<128xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<512x128xf32>, %arg36: memref<512xf32>, %arg37: memref<128x512xf32>, %arg38: memref<128xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128x128xf32>, %arg42: memref<128xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<f32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>) {
    %0 = memref.alloc() : memref<f32>
    %1 = memref.alloc() : memref<30522xf32>
    %2 = memref.alloc() : memref<512x128xf32>
    %3 = memref.alloc() : memref<128x128xf32>
    %4 = memref.alloc() : memref<2x128xf32>
    %5 = memref.alloc() : memref<30522x128xf32>
    %6 = memref.alloc() : memref<128xf32>
    %7 = memref.alloc() : memref<128xf32>
    %8 = memref.alloc() : memref<2x128x128xf32>
    %9 = memref.alloc() : memref<128xf32>
    %10 = memref.alloc() : memref<128x128xf32>
    %11 = memref.alloc() : memref<2x128x128xf32>
    %12 = memref.alloc() : memref<128xf32>
    %13 = memref.alloc() : memref<128x128xf32>
    %14 = memref.alloc() : memref<2x128x128xf32>
    %15 = memref.alloc() : memref<128xf32>
    %16 = memref.alloc() : memref<128x128xf32>
    %17 = memref.alloc() : memref<2x128x128xf32>
    %18 = memref.alloc() : memref<2x2x128x64xf32>
    %19 = memref.alloc() : memref<2x2x128x64xf32>
    %20 = memref.alloc() : memref<2x2x128x128xf32>
    %21 = memref.alloc() : memref<2x2x128x64xf32>
    %22 = memref.alloc() : memref<2x2x128x128xf32>
    %23 = memref.alloc() : memref<2x2x128x64xf32>
    %24 = memref.alloc() : memref<2x128x2x64xf32>
    %25 = memref.alloc() : memref<128xf32>
    %26 = memref.alloc() : memref<128x128xf32>
    %27 = memref.alloc() : memref<2x128x128xf32>
    %28 = memref.alloc() : memref<2x128x128xf32>
    %29 = memref.alloc() : memref<128xf32>
    %30 = memref.alloc() : memref<128xf32>
    %31 = memref.alloc() : memref<2x128x128xf32>
    %32 = memref.alloc() : memref<512xf32>
    %33 = memref.alloc() : memref<512x128xf32>
    %34 = memref.alloc() : memref<2x128x128xf32>
    %35 = memref.alloc() : memref<128xf32>
    %36 = memref.alloc() : memref<128x512xf32>
    %37 = memref.alloc() : memref<2x128x512xf32>
    %38 = memref.alloc() : memref<2x128x128xf32>
    %39 = memref.alloc() : memref<128xf32>
    %40 = memref.alloc() : memref<128xf32>
    %41 = memref.alloc() : memref<2x128x128xf32>
    %42 = memref.alloc() : memref<128xf32>
    %43 = memref.alloc() : memref<128x128xf32>
    %44 = memref.alloc() : memref<2x128x128xf32>
    %45 = memref.alloc() : memref<128xf32>
    %46 = memref.alloc() : memref<128x128xf32>
    %47 = memref.alloc() : memref<2x128x128xf32>
    %48 = memref.alloc() : memref<128xf32>
    %49 = memref.alloc() : memref<128x128xf32>
    %50 = memref.alloc() : memref<2x128x128xf32>
    %51 = memref.alloc() : memref<2x2x128x64xf32>
    %52 = memref.alloc() : memref<2x2x128x64xf32>
    %53 = memref.alloc() : memref<2x2x128x128xf32>
    %54 = memref.alloc() : memref<2x2x128x64xf32>
    %55 = memref.alloc() : memref<2x2x128x128xf32>
    %56 = memref.alloc() : memref<2x2x128x64xf32>
    %57 = memref.alloc() : memref<2x128x2x64xf32>
    %58 = memref.alloc() : memref<128xf32>
    %59 = memref.alloc() : memref<128x128xf32>
    %60 = memref.alloc() : memref<2x128x128xf32>
    %61 = memref.alloc() : memref<2x128x128xf32>
    %62 = memref.alloc() : memref<128xf32>
    %63 = memref.alloc() : memref<128xf32>
    %64 = memref.alloc() : memref<2x128x128xf32>
    %65 = memref.alloc() : memref<512xf32>
    %66 = memref.alloc() : memref<512x128xf32>
    %67 = memref.alloc() : memref<2x128x128xf32>
    %68 = memref.alloc() : memref<128xf32>
    %69 = memref.alloc() : memref<128x512xf32>
    %70 = memref.alloc() : memref<2x128x512xf32>
    %71 = memref.alloc() : memref<2x128x128xf32>
    %72 = memref.alloc() : memref<128xf32>
    %73 = memref.alloc() : memref<128xf32>
    %74 = memref.alloc() : memref<2x128x128xf32>
    %75 = memref.alloc() : memref<128xf32>
    %76 = memref.alloc() : memref<128x128xf32>
    %77 = memref.alloc() : memref<2x128x128xf32>
    %78 = memref.alloc() : memref<128xf32>
    %79 = memref.alloc() : memref<128xf32>
    %80 = memref.alloc() : memref<2x128x128xf32>
    %81 = memref.alloc() : memref<2x128x128xf32>
    %82 = memref.alloc() : memref<256x128xf32>
    %83 = memref.alloc() : memref<f32>
    %84 = memref.alloc() : memref<256xf32>
    %85 = memref.alloc() : memref<f32>
    %86 = memref.alloc() : memref<f32>
    %87 = memref.alloc() : memref<256xf32>
    %88 = memref.alloc() : memref<256xf32>
    %89 = memref.alloc() : memref<256x30522xf32>
    %90 = memref.alloc() : memref<256x128xf32>
    %91 = memref.alloc() : memref<256xf32>
    %92 = memref.alloc() : memref<256xf32>
    %93 = memref.alloc() : memref<2x128x128xf32>
    %94 = memref.alloc() : memref<0xf32>
    %95 = memref.alloc() : memref<2x128x128xf32>
    %96 = memref.alloc() : memref<2x128x128xf32>
    %97 = memref.alloc() : memref<2x128x128xf32>
    %98 = memref.alloc() : memref<256xf32>
    %99 = memref.alloc() : memref<256xf32>
    %100 = memref.alloc() : memref<2x128x128xf32>
    %101 = memref.alloc() : memref<2x128x128xf32>
    %102 = memref.alloc() : memref<0xf32>
    %103 = memref.alloc() : memref<2x128x512xf32>
    %104 = memref.alloc() : memref<2x128x512xf32>
    %105 = memref.alloc() : memref<2x128x128xf32>
    %106 = memref.alloc() : memref<256xf32>
    %107 = memref.alloc() : memref<256xf32>
    %108 = memref.alloc() : memref<2x128x128xf32>
    %109 = memref.alloc() : memref<2x128x128xf32>
    %110 = memref.alloc() : memref<2x128x128xf32>
    %111 = memref.alloc() : memref<2x128x2x64xf32>
    %112 = memref.alloc() : memref<2x2x128x64xf32>
    %113 = memref.alloc() : memref<2x2x128x64xf32>
    %114 = memref.alloc() : memref<2x2x128x128xui8>
    %115 = memref.alloc() : memref<2x2x128x128xf32>
    %116 = memref.alloc() : memref<2x2x128x128xf32>
    %117 = memref.alloc() : memref<2x2x128x128xf32>
    %118 = memref.alloc() : memref<2x2x128x64xf32>
    %119 = memref.alloc() : memref<2x2x128x64xf32>
    %120 = memref.alloc() : memref<2x128x128xf32>
    %121 = memref.alloc() : memref<256xf32>
    %122 = memref.alloc() : memref<256xf32>
    %123 = memref.alloc() : memref<2x128x128xf32>
    %124 = memref.alloc() : memref<2x128x128xf32>
    %125 = memref.alloc() : memref<0xf32>
    %126 = memref.alloc() : memref<2x128x512xf32>
    %127 = memref.alloc() : memref<2x128x512xf32>
    %128 = memref.alloc() : memref<2x128x128xf32>
    %129 = memref.alloc() : memref<256xf32>
    %130 = memref.alloc() : memref<256xf32>
    %131 = memref.alloc() : memref<2x128x128xf32>
    %132 = memref.alloc() : memref<2x128x128xf32>
    %133 = memref.alloc() : memref<2x128x128xf32>
    %134 = memref.alloc() : memref<2x128x2x64xf32>
    %135 = memref.alloc() : memref<2x2x128x64xf32>
    %136 = memref.alloc() : memref<2x2x128x64xf32>
    %137 = memref.alloc() : memref<2x2x128x128xui8>
    %138 = memref.alloc() : memref<2x2x128x128xf32>
    %139 = memref.alloc() : memref<2x2x128x128xf32>
    %140 = memref.alloc() : memref<2x2x128x128xf32>
    %141 = memref.alloc() : memref<2x2x128x64xf32>
    %142 = memref.alloc() : memref<2x2x128x64xf32>
    %143 = memref.alloc() : memref<256xf32>
    %144 = memref.alloc() : memref<256xf32>
    %145 = memref.alloc() : memref<2x128x128xf32>
    %146 = memref.alloc() : memref<128x128xf32>
    %147 = memref.alloc() : memref<256x128xf32>
    %148 = memref.alloc() : memref<256x128xf32>
    %149 = memref.alloc() : memref<1x128xi64>
    %150 = memref.alloc() : memref<128xi64>
    %151 = memref.alloc() : memref<1x128xi64>
    %152 = memref.alloc() : memref<2x128x128xf32>
    %153 = memref.alloc() : memref<f32>
    %154 = memref.alloc() : memref<2x128xf32>
    %155 = memref.alloc() : memref<512x128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    "lmhlo.constant"(%155) {value = dense<0.000000e+00> : tensor<512x128xf32>} : (memref<512x128xf32>) -> ()
    "lmhlo.constant"(%154) {value = dense<0.000000e+00> : tensor<2x128xf32>} : (memref<2x128xf32>) -> ()
    "lmhlo.constant"(%153) {value = dense<0xFF800000> : tensor<f32>} : (memref<f32>) -> ()
    "lmhlo.constant"(%152) {value = dense<-0.000000e+00> : tensor<2x128x128xf32>} : (memref<2x128x128xf32>) -> ()
    "lmhlo.slice"(%arg2, %151) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    "lmhlo.reshape"(%151, %150) : (memref<1x128xi64>, memref<128xi64>) -> ()
    "lmhlo.slice"(%arg3, %149) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %156:2 = call @Unknown0(%arg1) : (memref<2x128xi64>) -> (memref<256xi64>, memref<256xi1>)
    %157:3 = call @Unknown1(%arg0) : (memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    "lmhlo.gather"(%arg4, %157#0, %148) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %158:3 = call @Unknown2(%150) : (memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    "lmhlo.gather"(%arg5, %158#0, %147) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %159:3 = call @Unknown3(%149) : (memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>)
    "lmhlo.gather"(%arg6, %159#0, %146) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>) -> ()
    %160 = call @Unknown4(%148, %147, %146) : (memref<256x128xf32>, memref<256x128xf32>, memref<128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%160, %arg7, %arg8, %145, %144, %143) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.custom_call"(%145, %arg9, %arg10, %142) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%145, %arg11, %arg12, %141) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%142, %141, %140) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%140, %152, %139, %138, %137) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%145, %arg13, %arg14, %136) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%139, %136, %135) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%135, %134) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.reshape"(%134, %133) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%133, %arg15, %arg16, %132) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%132, %arg17, %arg18, %145, %131, %130, %129, %128) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%131, %arg19, %arg20, %127, %126, %125) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%127, %arg21, %arg22, %124) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%124, %arg23, %arg24, %131, %123, %122, %121, %120) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%123, %arg25, %arg26, %119) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%123, %arg27, %arg28, %118) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%119, %118, %117) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%117, %152, %116, %115, %114) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%123, %arg29, %arg30, %113) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%116, %113, %112) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%112, %111) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.reshape"(%111, %110) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%110, %arg31, %arg32, %109) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%109, %arg33, %arg34, %123, %108, %107, %106, %105) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%108, %arg35, %arg36, %104, %103, %102) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%104, %arg37, %arg38, %101) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%101, %arg39, %arg40, %108, %100, %99, %98, %97) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%100, %arg41, %arg42, %96, %95, %94) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%96, %arg43, %arg44, %93, %92, %91) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.reshape"(%93, %90) : (memref<2x128x128xf32>, memref<256x128xf32>) -> ()
    "lmhlo.dot"(%90, %arg4, %89) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>) -> ()
    %161:2 = call @Unknown5(%89, %arg45) : (memref<256x30522xf32>, memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<256x30522xf32>)
    "lmhlo.reduce"(%161#1, %153, %88) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.maximum"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %162:2 = call @Unknown6(%88, %161#1) : (memref<256xf32>, memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<256x30522xf32>)
    "lmhlo.reduce"(%162#1, %0, %87) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %163 = call @Unknown7(%87) : (memref<256xf32>) -> memref<256xf32>
    %164:4 = call @Unknown8(%163, %162#0, %156#0, %156#1) : (memref<256xf32>, memref<256x30522xf32>, memref<256xi64>, memref<256xi1>) -> (memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>)
    "lmhlo.reduce"(%164#0, %0, %86) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    "lmhlo.reduce"(%164#0, %0, %85) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %165 = call @Unknown9(%85) : (memref<f32>) -> memref<f32>
    %166 = call @Unknown10(%165, %164#2) : (memref<f32>, memref<256x30522xf32>) -> memref<256x30522xf32>
    "lmhlo.reduce"(%166, %0, %84) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    "lmhlo.reduce"(%164#1, %0, %83) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %167 = call @Unknown11(%83, %86) : (memref<f32>, memref<f32>) -> memref<f32>
    %168:2 = call @Unknown12(%84, %164#3, %166) : (memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<2x128x30522xf32>)
    %169 = call @MatmulOp0(%90, %168#0) : (memref<256x128xf32>, memref<256x30522xf32>) -> memref<30522x128xf32>
    "lmhlo.dot"(%168#0, %arg4, %82) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>) -> ()
    "lmhlo.reshape"(%82, %81) : (memref<256x128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%81, %96, %arg43, %92, %91, %80, %79, %78) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%80, %100, %arg41, %95, %94, %77, %76, %75) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%77, %97, %arg39, %99, %98, %74, %73, %72, %71) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%74, %104, %arg37, %70, %69, %68) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%70, %108, %arg35, %103, %102, %67, %66, %65) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %170 = call @Unknown13(%71, %67) : (memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%170, %105, %arg33, %107, %106, %64, %63, %62, %61) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%64, %110, %arg31, %60, %59, %58) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.reshape"(%60, %57) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.custom_call"(%57, %56) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%56, %116, %113, %55, %54) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%55, %116, %114, %53) {api_version = 1 : i32, backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%53, %119, %118, %52, %51) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%52, %123, %arg25, %50, %49, %48) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%54, %123, %arg29, %47, %46, %45) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%51, %123, %arg27, %44, %43, %42) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %171 = call @Unknown14(%61, %50, %47, %44) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%171, %120, %arg23, %122, %121, %41, %40, %39, %38) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%41, %127, %arg21, %37, %36, %35) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%37, %131, %arg19, %126, %125, %34, %33, %32) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %172 = call @Unknown15(%38, %34) : (memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%172, %128, %arg17, %130, %129, %31, %30, %29, %28) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%31, %133, %arg15, %27, %26, %25) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.reshape"(%27, %24) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.custom_call"(%24, %23) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%23, %139, %136, %22, %21) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%22, %139, %137, %20) {api_version = 1 : i32, backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%20, %142, %141, %19, %18) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%19, %145, %arg9, %17, %16, %15) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%21, %145, %arg13, %14, %13, %12) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%18, %145, %arg11, %11, %10, %9) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %173 = call @Unknown16(%28, %17, %14, %11) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%173, %160, %arg7, %144, %143, %8, %7, %6) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %174:2 = call @Unknown17(%157#2, %8, %158#2) : (memref<256xi1>, memref<2x128x128xf32>, memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>)
    "lmhlo.scatter"(%169, %157#1, %174#0, %5) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):  // no predecessors
      %176 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%176) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>) -> ()
    "lmhlo.scatter"(%154, %158#1, %174#1, %4) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):  // no predecessors
      %176 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%176) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>) -> ()
    "lmhlo.reduce"(%8, %0, %3) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<2x128x128xf32>, memref<f32>, memref<128x128xf32>) -> ()
    %175 = call @Unknown18(%159#2, %3) : (memref<128xi1>, memref<128x128xf32>) -> memref<128x128xf32>
    "lmhlo.scatter"(%155, %159#1, %175, %2) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):  // no predecessors
      %176 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%176) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>) -> ()
    "lmhlo.reduce"(%168#1, %0, %1) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<2x128x30522xf32>, memref<f32>, memref<30522xf32>) -> ()
    return %161#0, %167, %5, %4, %2, %7, %6, %16, %15, %10, %9, %13, %12, %26, %25, %30, %29, %33, %32, %36, %35, %40, %39, %49, %48, %43, %42, %46, %45, %59, %58, %63, %62, %66, %65, %69, %68, %73, %72, %76, %75, %79, %78, %1 : memref<2x128x30522xf32>, memref<f32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>
  }
}

