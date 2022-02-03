import numpy as np

from mlir import ir
from mlir.dialects import builtin
from mlir.dialects import std

def mlir_type_to_dtype(mlir_type):
    if str(mlir_type) == 'f64':
        return np.float64
    if str(mlir_type) == 'f32':
        return np.float32
    if str(mlir_type) == 'i64':
        return np.int64
    if str(mlir_type) == 'ui32':
        return np.uint32
    if str(mlir_type) == 'ui8':
        return np.uint8
    if str(mlir_type) == 'i1':
        return np.bool
    if str(mlir_type) == '!tf_type.string':
        return np.dtype('O')
    raise NotImplementedError("unsupported mlir type {}".format(mlir_type))

def mlir_clone_region(old_region, new_region, mapping):
    assert len(old_region.blocks) == 1, "only single block op is supported now"
    old_block = old_region.blocks[0]
    new_block = ir.Block.create_at_start(new_region, [i.type for i in old_block.arguments])

    for old_arg, new_arg in zip(old_block.arguments, new_block.arguments):
        mapping[old_arg] = new_arg

    for op in old_block:
        new_block.append(mlir_clone_op(op, mapping))

def mlir_clone_op(op, mapping):
    # don't support control flow op
    new_op = ir.Operation.create(
        op.operation.name,  # name
        [i.type for i in op.results], # result types
        [mapping.get(i, i) for i in op.operands], # operands
        {i.name: i.attr for i in op.attributes}, # attributes
        [], # successors
        len(op.regions) # nr_regions
    )

    for src, dst in zip(op.results, new_op.results):
        mapping[src] = dst

    for src, dst in zip(op.regions, new_op.regions):
        mlir_clone_region(src, dst, mapping)

    assert new_op.operation.verify()
    return new_op

def mlir_single_op_outlining(op):
    with op.context, op.location:
        module = ir.Module.create()

        with ir.InsertionPoint(module.body):
            f_type = ir.FunctionType.get(
                [i.type for i in op.operands],
                [i.type for i in op.results])
            f_op = builtin.FuncOp("main", f_type)
            entry_block = f_op.add_entry_block()
            with ir.InsertionPoint(entry_block):
                new_op = mlir_clone_op(op, dict(zip(op.operands, f_op.arguments)))
                std.ReturnOp(new_op.results)
            assert f_op.operation.verify()

        assert module.operation.verify()
        return module

def mlir_attr_to_pyobj(attribute):
    if ir.DictAttr.isinstance(attribute):
        dict_attr = ir.DictAttr(attribute)
        return { dict_attr[idx].name: mlir_attr_to_pyobj(dict_attr[idx].attr) for idx in range(len(dict_attr)) }

    for attr_type_name in ["StringAttr", "BoolAttr", "IntegerAttr", "FloatAttr"]:
        attr_type_cls = getattr(ir, attr_type_name)
        if attr_type_cls.isinstance(attribute):
            return attr_type_cls(attribute).value

    raise NotImplementedError("unsupported attribute {}".format(attribute))

