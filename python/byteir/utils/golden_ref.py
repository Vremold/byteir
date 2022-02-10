import logging
import numpy as np

from byteir.ir import IRExecutor, mlir_type_to_dtype
from mlir import ir
from mlir.dialects.mhlo import register_mhlo_dialect


class MhloGoldenRefGenerator:

    context = None
    module = None
    value2tensor = None

    def __init__(self, module_str):
        self.context = ir.Context()
        register_mhlo_dialect(self.context)
        self.context.allow_unregistered_dialects = True
        self.module = ir.Module.parse(module_str, self.context)
        self.value2tensor = {}

    def generate(self):
        func = self.module.body.operations[0]

        for i in func.arguments:
            shaped_type = ir.ShapedType(i.type)
            shape = shaped_type.shape
            dtype = mlir_type_to_dtype(shaped_type.element_type)
            if i in self.value2tensor:
                assert(shape == self.value2tensor[i].shape)
                assert(dtype == self.value2tensor[i].dtype)
                continue
            elif dtype is np.str:
                self.value2tensor[i] = ""
            else:
                self.value2tensor[i] = np.random.normal(
                    loc=0.0, scale=0.1, size=shape).astype(dtype)

        for op in func.entry_block:
            if op.operation.name == "std.return":
                continue
            inputs = list(self.value2tensor.get(i) for i in op.operands)
            outputs = IRExecutor.execute(op, inputs)
            self.value2tensor.update(dict(zip(op.results, outputs)))

    @classmethod
    def load_from_string(cls, module_str):
        return MhloGoldenRefGenerator(module_str)

    @classmethod
    def load_from_file(cls, module_path):
        with open(module_path, 'r') as f:
            return cls.load_from_string(f.read())
