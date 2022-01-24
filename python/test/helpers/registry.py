from byteir.ir import IRExecutor, MhloCustomCallExecutor, mlir_attr_to_pyobj
from mlir import ir

@IRExecutor.register("mhlo.add")
def _dispatch_add(op, inputs):
    return [inputs[0] + inputs[1],]

@MhloCustomCallExecutor.register("test_add")
def _dispatch_test_add(op, inputs):
    backend_config_attr = op.attributes["backend_config"]
    assert ir.StringAttr.isinstance(op.attributes["backend_config"])
    backend_config = ir.StringAttr(backend_config_attr).value
    op_attributes = mlir_attr_to_pyobj(ir.Attribute.parse(backend_config, op.context))
    return [inputs[0] + op_attributes['value'],]
