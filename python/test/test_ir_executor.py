import numpy as np

from byteir.ir import IRExecutor
from mlir import ir
from mlir.dialects.mhlo import register_mhlo_dialect

from helpers.registry import *

def test_add():
    with ir.Context() as context:
        register_mhlo_dialect(context)
        module = ir.Module.parse("""
            func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                %0 = mhlo.add %arg0, %arg1 : tensor<4xf32>
                return %0: tensor<4xf32>
            }""")
        x = np.random.random(size=(4)).astype(np.float32)
        y = np.random.random(size=(4)).astype(np.float32)
        out, = IRExecutor.execute(module.body.operations[0].entry_block.operations[0], [x, y])
        np.testing.assert_almost_equal(out, x + y)

def test_custom_call():
    with ir.Context() as context:
        register_mhlo_dialect(context)
        module = ir.Module.parse("""
            func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                %0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "{value = 1.000000e+00 : f32}", call_target_name = "test_add", has_side_effect = false} : (tensor<4xf32>) -> tensor<4xf32>
                return %0: tensor<4xf32>
            }""")
        x = np.random.random(size=(4)).astype(np.float32)
        out, = IRExecutor.execute(module.body.operations[0].entry_block.operations[0], [x,])
        np.testing.assert_almost_equal(out, x + 1.0)
