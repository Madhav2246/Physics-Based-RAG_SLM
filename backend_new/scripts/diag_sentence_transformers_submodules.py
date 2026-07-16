import sys

# Ensure torch is imported first to set DLL paths
import torch
import faiss

print("[Diag] Importing sentence_transformers.util...", flush=True)
import sentence_transformers.util
print("[Diag] sentence_transformers.util imported successfully.", flush=True)

print("[Diag] Importing sentence_transformers.models...", flush=True)
import sentence_transformers.models
print("[Diag] sentence_transformers.models imported successfully.", flush=True)

print("[Diag] Importing sentence_transformers.datasets...", flush=True)
import sentence_transformers.datasets
print("[Diag] sentence_transformers.datasets imported successfully.", flush=True)

print("[Diag] Importing sentence_transformers.losses...", flush=True)
import sentence_transformers.losses
print("[Diag] sentence_transformers.losses imported successfully.", flush=True)

print("[Diag] Importing sentence_transformers.evaluation...", flush=True)
import sentence_transformers.evaluation
print("[Diag] sentence_transformers.evaluation imported successfully.", flush=True)

print("[Diag] Importing sentence_transformers.SentenceTransformer...", flush=True)
import sentence_transformers.SentenceTransformer
print("[Diag] sentence_transformers.SentenceTransformer imported successfully.", flush=True)

print("[Diag] Success!", flush=True)
