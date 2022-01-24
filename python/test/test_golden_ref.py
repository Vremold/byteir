import numpy as np

from byteir.utils.golden_ref import MhloGoldenRefGenerator
from helpers.registry import *

def test_golden_ref():
    module_str = """
    func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
        %0 = mhlo.add %arg0, %arg1 : tensor<4xf32>
        %1 = "mhlo.custom_call"(%0) {api_version = 1 : i32, backend_config = "{value = 1.000000e+00 : f32}", call_target_name = "test_add", has_side_effect = false} : (tensor<4xf32>) -> tensor<4xf32>
        return %1: tensor<4xf32>
    }"""
    
    gen = MhloGoldenRefGenerator.load_from_string(module_str)
    gen.generate()
    module = gen.module
    func = module.body.operations[0]
    block = func.entry_block
    x = gen.value2tensor[func.arguments[0]]
    y = gen.value2tensor[func.arguments[1]]
    out0 = gen.value2tensor[block.operations[0].results[0]]
    out1 = gen.value2tensor[block.operations[1].results[0]]
    np.testing.assert_almost_equal(out0, x + y)
    np.testing.assert_almost_equal(out1, x + y + 1.0)

