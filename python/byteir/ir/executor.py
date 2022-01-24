from mlir import ir
from collections import namedtuple

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
    def execute(cls, op, inputs):
        # TODO: type check
        return cls.dispatch(op, inputs)


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
