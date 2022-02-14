from byteir.ir import IRExecutor, MhloCustomCallExecutor, mlir_attr_to_pyobj
from mlir import ir

@IRExecutor.register("mhlo.add")
def _dispatch_add(op, inputs):
    return [inputs[0] + inputs[1],]

@IRExecutor.register("mhlo.get_tuple_element")
def _dispatch_get_tuple_element(op, inputs):
    index_attr = op.attributes["index"]
    assert ir.IntegerAttr.isinstance(index_attr)
    index = ir.IntegerAttr(index_attr).value
    tuple_input = inputs[0]
    output = tuple_input[index]
    return [output,]

@MhloCustomCallExecutor.register("test_add")
def _dispatch_test_add(op, inputs):
    backend_config_attr = op.attributes["backend_config"]
    assert ir.StringAttr.isinstance(op.attributes["backend_config"])
    backend_config = ir.StringAttr(backend_config_attr).value
    op_attributes = mlir_attr_to_pyobj(ir.Attribute.parse(backend_config, op.context))
    return [inputs[0] + op_attributes['value'],]
