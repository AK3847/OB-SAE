"""Pre-download the base model weights so training does not stall on the network."""
import sys

from huggingface_hub import snapshot_download

repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-9B"
path = snapshot_download(
    repo,
    allow_patterns=["*.json", "*.txt", "*.jinja", "*.safetensors"],
    max_workers=4,
)
print(f"downloaded to {path}")
