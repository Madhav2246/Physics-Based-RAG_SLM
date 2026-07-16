import sys

print("[Diag] Importing torch...", flush=True)
import torch

print("[Diag] Importing transformers...", flush=True)
import transformers
print("[Diag] transformers imported successfully.", flush=True)

print("[Diag] Importing AutoModel from transformers...", flush=True)
from transformers import AutoModel
print("[Diag] AutoModel imported successfully!", flush=True)

print("[Diag] Success!", flush=True)
