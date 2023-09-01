import torch
from flash_attn import flash_attn_func

b = 1
seq_len = 32
num_heads = 3
head_dims = 8

input_len = b * seq_len * num_heads * head_dims
# q = torch.arange(input_len, dtype=torch.float16, device='cuda').reshape((1, 32, 3, 8))
q = torch.arange(input_len, dtype=torch.float16, device='cuda').reshape((1, 32, 3, 8))/3000
k = torch.arange(input_len, dtype=torch.float16, device='cuda').reshape((1, 32, 3, 8))/4000
v = torch.arange(input_len, dtype=torch.float16, device='cuda').reshape((1, 32, 3, 8))/5000

out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=0.5, causal=False)

print("generating ground truth for flash_attn fwd output...")
out = torch.flatten(out)
of = open("flash_attn_fwd_outputs.data", "w")
for num in out:
    of.write(f"{num} ")
of.close()
