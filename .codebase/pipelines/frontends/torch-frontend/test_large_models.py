import torch
import numpy as np

import os
import tempfile
from pathlib import Path

import torch_frontend
from torch_frontend import convert_to_mhlo_via_torch_mlir

from mhlo_tools.ir_executor import Interpreter
from mhlo_tools.ir_executor.helper import mlir_attr_to_pyobj

LARGE_MODEL_PATH = os.environ["TORCH_LARGE_MODEL_PATH"]
# (model_path, rtol, atol)
MODEL_LIST = [("pytorch/sar_relevance_cross_model_latest/28365.ts", 0.002, 0.0001), 
              ("pytorch/tt_label3_0607/torch_model_1654572315533.jit.revert.ts", 0.001, 0.0001),
              ("pytorch/swinv2_tiny/swinv2_tiny.pt", 0.001, 0.004), # FIXME(lyq): Max absolute difference: 0.003235, Max relative difference: 3.
              ("pytorch/rtc1/torch_jit_1682337197499.jit.revert", 0.001, 0.0001),
              ("pytorch/rtc/model.jit", 0.001, 0.0001),
              ("pytorch/moe_static/mm_mf_jingpai_12_1_moe_context_matx/torch_model_1660205850750.jit.3.revert", 0.05, 0.0001)]

os.environ['TORCH_JIT_DISABLE_NEW_EXECUTOR'] = '1'
torch._C._jit_set_nvfuser_enabled(False)

def reload_model(model):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tmp.jit"
        model.save(path)
        model = torch.jit.load(path)
    return model

def convert_to_mhlo_via_torch_mlir(jit_model_path, rtol, atol):
    dir = os.path.abspath(os.path.dirname(jit_model_path))
    ts_module = torch.jit.load(jit_model_path, map_location="cuda").eval()
    sample_inputs = torch.load(os.path.join(dir, "batch_sample_inputs"), map_location="cuda")
    torch_outputs = ts_module(*sample_inputs)
    print(torch_outputs)

    module = torch_frontend.convert_to_mhlo_via_torch_mlir(ts_module, sample_inputs)
    module_str = str(module.operation.get_asm(enable_debug_info=False))

    # run mhlo-tools and compare outputs' diff
    with Interpreter.load_from_string(module_str, True) as interp:
        mhlo_outputs = interp.call_function("forward", [i.detach().cpu().numpy() for i in sample_inputs])
        print(mhlo_outputs)
        if isinstance(torch_outputs, torch.Tensor):
            torch_outputs = [torch_outputs.detach().cpu().numpy()]
        else:
            torch_outputs = [i.detach().cpu().numpy() for i in torch_outputs]
        np.testing.assert_allclose(mhlo_outputs, torch_outputs, rtol=rtol, atol=atol)


def test_large_models():
    for model in MODEL_LIST:
        print(model[0])
        convert_to_mhlo_via_torch_mlir(os.path.join(LARGE_MODEL_PATH, model[0]), model[1], model[2])

if __name__ == "__main__":
    test_large_models()