import torch
from functorch import make_fx

import os
import tempfile
from pathlib import Path

import torch_frontend
from torch_frontend import convert_to_mhlo_via_torch_mlir

LARGE_MODEL_PATH = os.environ["TORCH_LARGE_MODEL_PATH"]
MODEL_LIST = ["sar_relevance_cross_model_latest/28365.ts", "tt_label3_0607/torch_model_1654572315533.jit.revert.ts"]

os.environ['TORCH_JIT_DISABLE_NEW_EXECUTOR'] = '1'

def fx_rewrite(ts_module, sample_inputs):
    torch._C._jit_set_nvfuser_enabled(False)
    fx_g = make_fx(ts_module)(*sample_inputs)
    for node in fx_g.graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten.add_.Tensor:
                node.target = torch.ops.aten.add.Tensor
            elif node.target == torch.ops.aten.sub_.Tensor:
                node.target = torch.ops.aten.sub.Tensor
            elif node.target == torch.ops.aten.mul_.Tensor:
                node.target = torch.ops.aten.mul.Tensor
            elif node.target == torch.ops.aten.div_.Tensor:
                node.target = torch.ops.aten.div.Tensor
    fx_g.graph.lint()
    fx_g.recompile()
    ts_module = torch.jit.trace(fx_g, sample_inputs)
    return ts_module

def reload_model(model):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tmp.jit"
        model.save(path)
        model = torch.jit.load(path)
    return model

def convert_to_mhlo_via_torch_mlir(jit_model_path):
    dir = os.path.abspath(os.path.dirname(jit_model_path))
    ts_module = torch.jit.load(jit_model_path, map_location="cuda")
    ts_module = ts_module.eval()
    sample_inputs = torch.load(os.path.join(dir, "batch_sample_inputs"), map_location="cuda")

    torch_frontend.register_decomposition_in_torchscript()
    torch._C._jit_pass_inline(ts_module.graph)
    torch._C._jit_pass_run_decompositions(ts_module.graph)
    ts_module = reload_model(ts_module).eval()

    ts_module = fx_rewrite(ts_module, sample_inputs)
    ts_module = reload_model(ts_module).eval()

    module = torch_frontend.convert_to_mhlo_via_torch_mlir(ts_module, sample_inputs)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))


def test_large_models():
    for model in MODEL_LIST:
        convert_to_mhlo_via_torch_mlir(os.path.join(LARGE_MODEL_PATH, model))

if __name__ == "__main__":
    test_large_models()