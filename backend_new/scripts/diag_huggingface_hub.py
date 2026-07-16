import sys

print("[Diag] Importing huggingface_hub...", flush=True)
import huggingface_hub
print("[Diag] huggingface_hub imported successfully.", flush=True)

print("[Diag] Checking huggingface_hub constants...", flush=True)
from huggingface_hub.constants import HF_HUB_DISABLE_SYMLINKS_WARNING
print(f"[Diag] Constant value: {HF_HUB_DISABLE_SYMLINKS_WARNING}", flush=True)

print("[Diag] Success!", flush=True)
