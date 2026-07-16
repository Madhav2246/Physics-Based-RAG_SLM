import os, sentence_transformers

init_path = sentence_transformers.__file__
print(f"[Diag] __init__.py path: {init_path}", flush=True)

with open(init_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print("[Diag] Printing first 50 lines of sentence_transformers/__init__.py:", flush=True)
for i, line in enumerate(lines[:50]):
    print(f"{i+1:02d}: {line.rstrip()}", flush=True)
