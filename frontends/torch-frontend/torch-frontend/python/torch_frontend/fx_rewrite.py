import torch
from .fx_tracer import HFTracer
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Union,
)

import torch.nn.functional as F

# GPT2 Attention patterns
def AttnPattern(query, key, value, causal_mask, mask_value, inv_scale, device, dropout_p):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    attn_weights = attn_weights / torch.full(
        [], inv_scale, dtype=torch.float16, device=device
    )
    attn_weights = torch.where(causal_mask, attn_weights.to(torch.float16), mask_value)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.type(torch.float16)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def AttnReplacement(q, k, v, causal_mask, mask_value, inv_scale, device, dropout_p):
    return torch.ops.aten.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=True,
        scale=1.0 / inv_scale
    )

# NanoGPT Attention patterns
def AttnPattern1(q, k, v, causal_mask, mask_value, scale, dropout_p):
    att = (q @ k.transpose(-2, -1)) * scale
    att = att.masked_fill(causal_mask, mask_value)
    att = F.softmax(att, dim=-1)
    att = torch.nn.functional.dropout(att, p=dropout_p)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    return y


def AttnReplacement1(q, k, v, causal_mask, mask_value, scale, dropout_p):
    return torch.ops.aten.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=True,
        scale=scale
    )


def canonicalize_graph_before_replacement(gm):
    for n in gm.graph.nodes:
        if n.op == "call_module":
            submod = gm.get_submodule(n.target)
            if isinstance(submod, torch.nn.Dropout):
                with gm.graph.inserting_before(n):
                    new_node = gm.graph.call_function(torch.nn.functional.dropout, args=n.args, kwargs={'p': submod.p, 'training': submod.training, 'inplace': submod.inplace})
                    n.replace_all_uses_with(new_node)
                    gm.graph.erase_node(n)
        if n.op == "call_function":
            if n.target == torch.nn.functional.softmax:
                new_args = dict(n.kwargs)
                if '_stacklevel' not in new_args:
                    new_args['_stacklevel'] = 3
                if 'dtype' not in new_args:
                    new_args['dtype'] = None                 
                n.kwargs = new_args
    gm.graph.lint()
    gm.recompile()
    return gm


# HuggingFace symbolic trace
# FIXME: workaround to trace torch.full
def hf_symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = None,
) -> torch.fx.GraphModule:
    tracer = HFTracer()
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return torch.fx.GraphModule(tracer.root, graph, name)

def fx_replace_attn_pattern(gm: torch.fx.GraphModule):
    gm = canonicalize_graph_before_replacement(gm)
    # Need hf_symbolic_trace to trace torch.full
    torch.fx.replace_pattern(gm, hf_symbolic_trace(AttnPattern), AttnReplacement)
    torch.fx.replace_pattern(gm, AttnPattern1, AttnReplacement1)
    return gm
