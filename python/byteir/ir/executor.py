import numpy as np

from mlir import ir
from collections import namedtuple
from .helper import mlir_type_to_dtype

class DispatchableIRExecutorMeta(type):
    def __new__(cls, name, bases, attrs):
        attrs['_dispatchers'] = []
        return super().__new__(cls, name, bases, attrs)


class DispatchableIRExecutorBase(metaclass=DispatchableIRExecutorMeta):
    DispatchTableEntry = namedtuple('DispatchTableEntry', ['matcher', 'fn', 'priority'])

    @classmethod
    def register(cls, matcher, fn, priority=0):
        cls._dispatchers.append(
            DispatchableIRExecutorBase.DispatchTableEntry(matcher, fn, priority))
        cls._dispatchers.sort(
            reverse=True, key=lambda entry: entry.priority)

    @classmethod
    def dispatch(cls, op, inputs):
        for entry in cls._dispatchers:
            if entry.matcher(op):
                return entry.fn(op, inputs)       

        raise NotImplementedError("unsupported operation {}".format(op))


class IRExecutor(DispatchableIRExecutorBase):
    @classmethod
    def register(cls, dialect):
        def matcher(op):
            return str(op.operation.name).startswith(dialect)

        def impl(fn):
            super(IRExecutor, cls).register(matcher, fn, len(dialect))
            return fn
        return impl

    @classmethod
    def _check_type_single(cls, op, ir_type, tensor):
        if ir.TupleType.isinstance(ir_type):
            assert isinstance(tensor, tuple), "expect tuple but got {}, associated op is {}".format(tensor, op)
            tuple_type = ir.TupleType(ir_type)
            sub_types = [tuple_type.get_type(i) for i in range(tuple_type.num_types)]
            sub_tensors = list(tensor)
            cls._check_types(op, sub_types, sub_tensors)
        else:
            assert isinstance(tensor, np.ndarray), "expect numpy.ndarray but got {}, associated op is {}".format(tensor, op)
            assert ir.ShapedType.isinstance(ir_type), "expect ShapedType but got {}, associated op is {}".format(ir_type, op)
            shaped_type = ir.ShapedType(ir_type)

            value_shape = list(shaped_type.shape)
            tensor_shape = list(tensor.shape)
            assert value_shape == tensor_shape, "shape mismatch {} vs {}, associated op is {}".format(value_shape, tensor_shape, op)

            if tensor.dtype != np.object: # tensorflow treat some of dtype as opaque object (i.e. np.str), ignore them when type checking
                value_dtype = mlir_type_to_dtype(shaped_type.element_type)
                tensor_dtype = tensor.dtype
                assert value_dtype == tensor_dtype, "dtype mismatch {} vs {}, associated op is {}".format(value_dtype, tensor_dtype, op)

    @classmethod
    def _check_types(cls, op, ir_types, tensors):
        assert isinstance(tensors, (tuple, list)), "the input tensors or output tensors of IRExecutor should be list or tuple"
        assert len(ir_types) == len(tensors), "the number of ir types and computing tensors mismatch, {} vs {}".format(ir_types, tensors)
        for ir_type, tensor in zip(ir_types, tensors):
            cls._check_type_single(op, ir_type, tensor)

    @classmethod
    def execute(cls, op, inputs):
        cls._check_types(op, [i.type for i in op.operands], inputs)
        outputs = cls.dispatch(op, inputs)
        cls._check_types(op, [i.type for i in op.results], outputs)
        return outputs


class MhloCustomCallExecutor(DispatchableIRExecutorBase):
    @classmethod
    def register(cls, name):
        def matcher(op):
            target_name_attr = op.attributes["call_target_name"]
            assert ir.StringAttr.isinstance(op.attributes["call_target_name"])
            target_name = ir.StringAttr(target_name_attr).value
            return str(target_name).startswith(name)

        def impl(fn):
            super(MhloCustomCallExecutor, cls).register(matcher, fn, len(name))
            return fn
        return impl


@IRExecutor.register("mhlo.custom_call")
def _dispatch_mhlo_custom_call(op, inputs):
    return MhloCustomCallExecutor.dispatch(op, inputs)
