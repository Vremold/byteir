import numpy as np

from byteir.ir import IRExecutor
from mlir import ir
from mlir.dialects.mhlo import register_mhlo_dialect

from helpers.registry import *

MODULE_STR = {
    "ADD": """
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4xf32>
    return %0: tensor<4xf32>
}""",

    "CUSTOM-CALL": """
func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "{value = 1.000000e+00 : f32}", call_target_name = "test_add", has_side_effect = false} : (tensor<4xf32>) -> tensor<4xf32>
    return %0: tensor<4xf32>
}""",

    "GET-TUPLE-ELEMENT": """
func.func @main(%arg0: tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32> {
    %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
    return %0: tensor<4xf32>
}""",
}

def test_add():
    with ir.Context() as context:
        register_mhlo_dialect(context)
        module = ir.Module.parse(MODULE_STR["ADD"])
        x = np.random.random(size=(4)).astype(np.float32)
        y = np.random.random(size=(4)).astype(np.float32)
        out, = IRExecutor.execute(module.body.operations[0].entry_block.operations[0], [x, y])
        np.testing.assert_almost_equal(out, x + y)

def test_get_tuple_element():
    with ir.Context() as context:
        register_mhlo_dialect(context)
        module = ir.Module.parse(MODULE_STR["GET-TUPLE-ELEMENT"])
        x = np.random.random(size=(4)).astype(np.float32)
        y = np.random.random(size=(4)).astype(np.float32)
        out, = IRExecutor.execute(module.body.operations[0].entry_block.operations[0], [(x, y)])
        np.testing.assert_almost_equal(out, x)

def test_custom_call():
    with ir.Context() as context:
        register_mhlo_dialect(context)
        module = ir.Module.parse(MODULE_STR["CUSTOM-CALL"])
        x = np.random.random(size=(4)).astype(np.float32)
        out, = IRExecutor.execute(module.body.operations[0].entry_block.operations[0], [x,])
        np.testing.assert_almost_equal(out, x + 1.0)

def test_type_checking():
    # check non-ndarray input
    with ir.Context() as context:
        register_mhlo_dialect(context)
        module = ir.Module.parse(MODULE_STR["ADD"])
        x = 1
        y = np.random.random(size=(5)).astype(np.float32)
        with np.testing.assert_raises_regex(AssertionError, "expect numpy.ndarray"):
            IRExecutor.execute(module.body.operations[0].entry_block.operations[0], [x, y])

    # check shape mismatch
    with ir.Context() as context:
        register_mhlo_dialect(context)
        module = ir.Module.parse(MODULE_STR["ADD"])
        x = np.random.random(size=(5)).astype(np.float32)
        y = np.random.random(size=(5)).astype(np.float32)
        with np.testing.assert_raises_regex(AssertionError, "shape mismatch"):
            IRExecutor.execute(module.body.operations[0].entry_block.operations[0], [x, y])

    # check dtype mismatch
    with ir.Context() as context:
        register_mhlo_dialect(context)
        module = ir.Module.parse(MODULE_STR["ADD"])
        x = np.random.random(size=(4)).astype(np.float64)
        y = np.random.random(size=(4)).astype(np.float64)
        with np.testing.assert_raises_regex(AssertionError, "dtype mismatch"):
            IRExecutor.execute(module.body.operations[0].entry_block.operations[0], [x, y])

    # check packed tuple input
    with ir.Context() as context:
        register_mhlo_dialect(context)
        module = ir.Module.parse(MODULE_STR['GET-TUPLE-ELEMENT'])
        x = np.random.random(size=(4)).astype(np.float32)
        y = np.random.random(size=(4)).astype(np.float32)
        with np.testing.assert_raises_regex(AssertionError, "the number of ir types and computing tensors mismatch"):
            out, = IRExecutor.execute(module.body.operations[0].entry_block.operations[0], [x, y]) # which should be [(x, y)]

    # check tuple element with wrong shape
    with ir.Context() as context:
        register_mhlo_dialect(context)
        module = ir.Module.parse(MODULE_STR['GET-TUPLE-ELEMENT'])
        x = np.random.random(size=(4)).astype(np.float32)
        y = np.random.random(size=(5)).astype(np.float32)
        with np.testing.assert_raises_regex(AssertionError, "shape mismatch"):
            out, = IRExecutor.execute(module.body.operations[0].entry_block.operations[0], [(x, y)]) 
