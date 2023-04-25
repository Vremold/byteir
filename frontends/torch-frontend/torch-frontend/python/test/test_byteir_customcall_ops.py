import torch
import torch as tu

import torch_frontend
from torch_frontend import convert_to_mhlo_via_torch_mlir

def custom_test_helper(module, inputs, custom_op_name):
    mlir_module = convert_to_mhlo_via_torch_mlir(module, inputs)
    mlir_str = mlir_module.operation.get_asm(large_elements_limit=10, enable_debug_info=False)
    compare_str = "mhlo.custom_call @{}".format(custom_op_name)
    assert compare_str in mlir_str
    # print(mlir_str)

# ==============================================================================

class NativeLayerNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias):
        list = [2, 2, 3]
        return torch.ops.aten.native_layer_norm(
            x, list, weight, bias, eps=0.5)

def test_native_layer_norm():
    inputs = [tu.rand(2, 5, 2, 2, 3).to(torch.float16), tu.rand(2, 2, 3).to(torch.float16), tu.rand(2, 2, 3).to(torch.float16)]
    custom_test_helper(NativeLayerNormModule(), inputs, "byteir.layer_norm")

# ==============================================================================

class OneHotModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.one_hot(x, num_classes=5)

def test_one_hot():
    inputs = [tu.arange(0, 5).long()]
    custom_test_helper(OneHotModule(), inputs, "byteir.one_hot")

# ==============================================================================

class TopKModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.topk(x, 3, dim=1)

def test_topk():
    inputs = [tu.randn(3, 4)]
    custom_test_helper(TopKModule(), inputs, "byteir.top_k")

# ==============================================================================

#TODO(lyq): add more tests
