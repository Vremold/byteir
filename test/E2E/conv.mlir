// RUN: byteir-opt -fuse-conv-bias-act -fusion-outlining -convert-hlo-to-lhlo -cse -convert-to-byre %s | FileCheck %s

func @conv_bias_act(%arg0: tensor<5x69x31x95xf32> {__placeholder__byre.argname = "A"}, %arg1: tensor<64x69x1x1xf32> {__placeholder__byre.argname = "B"}, %arg2: tensor<64xf32> {__placeholder__byre.argname = "C"}) -> (tensor<5x64x31x95xf32> {__placeholder__byre.argname = "D"}) attributes {__placeholder__byre.entry_point} {
    %1 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<5x69x31x95xf32>, tensor<64x69x1x1xf32>) -> tensor<5x64x31x95xf32>
    %2 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<5x64x31x95xf32>
    %3 = mhlo.add %1, %2 : tensor<5x64x31x95xf32>
    %4 = "ace.activate"(%3) {act_func = "relu"} : (tensor<5x64x31x95xf32>) -> tensor<5x64x31x95xf32>
    return %4 : tensor<5x64x31x95xf32>
}
// CHECK-LABEL: func @conv_bias_act
// CHECK:  byre.compute @ConvBiasOp{{.*}}{act_func = "relu", batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>}
