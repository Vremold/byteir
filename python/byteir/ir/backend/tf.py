from ..executor import IRExecutor
from ..helper import mlir_single_op_outlining

from tf_mlir_ext import parse_and_evaluate_simple_module

@IRExecutor.register("tf.")
def _dispatch_tf(op, inputs):
    tf_module = mlir_single_op_outlining(op)
    outputs = parse_and_evaluate_simple_module(str(tf_module), inputs)
    return outputs
