import os
# Import these first to prevent top-level sentence_transformers import crash
import numpy
import scipy
import sympy
import torch
import faiss
import sentence_transformers

st_dir = os.path.dirname(sentence_transformers.__file__)
util_path = os.path.join(st_dir, "util.py")
print(f"[Diag] util.py path: {util_path}", flush=True)

with open(util_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print("[Diag] Printing first 60 lines of util.py:", flush=True)
for i, line in enumerate(lines[:60]):
    print(f"{i+1:02d}: {line.rstrip()}", flush=True)
