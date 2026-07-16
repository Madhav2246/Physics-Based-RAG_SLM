"""
diag_imports.py
---------------
Diagnostics script to find which dependency is causing silent crashes/segfaults.
"""
import sys, os
os.environ["HF_HOME"] = "d:/S6/NLP/Physics_Based_RAG_SLM/hf_cache"

print("[Diag] Starting diagnostic imports...", flush=True)

try:
    print("[Diag] 1. Importing numpy...", end="", flush=True)
    import numpy as np
    print(" OK", flush=True)
except Exception as e:
    print(f" FAIL: {e}", flush=True)

try:
    print("[Diag] 2. Importing scipy...", end="", flush=True)
    import scipy
    print(" OK", flush=True)
except Exception as e:
    print(f" FAIL: {e}", flush=True)

try:
    print("[Diag] 3. Importing sympy...", end="", flush=True)
    import sympy
    print(" OK", flush=True)
except Exception as e:
    print(f" FAIL: {e}", flush=True)

try:
    print("[Diag] 4. Importing torch...", end="", flush=True)
    import torch
    print(" OK", flush=True)
    print(f"       CUDA Available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"       CUDA Device: {torch.cuda.get_device_name(0)}", flush=True)
except Exception as e:
    print(f" FAIL: {e}", flush=True)

try:
    print("[Diag] 5. Importing faiss...", end="", flush=True)
    import faiss
    print(" OK", flush=True)
except Exception as e:
    print(f" FAIL: {e}", flush=True)

try:
    print("[Diag] 6. Importing sentence_transformers...", end="", flush=True)
    import sentence_transformers
    print(" OK", flush=True)
except Exception as e:
    print(f" FAIL: {e}", flush=True)

try:
    print("[Diag] 7. Importing openai...", end="", flush=True)
    import openai
    print(" OK", flush=True)
except Exception as e:
    print(f" FAIL: {e}", flush=True)

print("[Diag] All core imports completed successfully!", flush=True)
