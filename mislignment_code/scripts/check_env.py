"""Verify the local environment can actually train on this GPU before downloading 19 GB of weights."""
import torch
import transformers
import peft
import trl
import bitsandbytes
import accelerate

print(f"torch         {torch.__version__}")
print(f"cuda available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device        {props.name}")
    print(f"capability    sm_{cap[0]}{cap[1]}")
    print(f"vram          {props.total_memory / 1024 ** 3:.2f} GiB")
    print(f"torch archs   {torch.cuda.get_arch_list()}")
    supported = f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list()
    print(f"arch in build {supported}")
print(f"transformers  {transformers.__version__}")
print(f"peft          {peft.__version__}")
print(f"trl           {trl.__version__}")
print(f"bitsandbytes  {bitsandbytes.__version__}")
print(f"accelerate    {accelerate.__version__}")

print(f"qwen3_5 known {'qwen3_5' in transformers.models.auto.configuration_auto.CONFIG_MAPPING_NAMES}")

if torch.cuda.is_available():
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print(f"bf16 matmul   ok ({(a @ a).float().abs().sum().item():.3e})")

    import bitsandbytes.nn as bnn
    layer = bnn.Linear4bit(512, 512, bias=False, compute_dtype=torch.bfloat16).cuda()
    out = layer(torch.randn(4, 512, device="cuda", dtype=torch.bfloat16))
    print(f"bnb 4bit fwd  ok {tuple(out.shape)}")
