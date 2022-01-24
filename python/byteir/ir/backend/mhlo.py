from ..executor import MhloCustomCallExecutor, IRExecutor
from ..helper import mlir_single_op_outlining

from tf_mlir_ext import parse_and_evaluate_simple_module

@IRExecutor.register("mhlo.")
def _dispatch_mhlo(op, inputs):
    mhlo_module = mlir_single_op_outlining(op)
    outputs = parse_and_evaluate_simple_module(str(mhlo_module), inputs)
    return outputs
