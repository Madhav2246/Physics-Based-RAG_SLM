import sys

print("[Diag] Importing torch first...", flush=True)
import torch
print("[Diag] torch imported successfully.", flush=True)

print("[Diag] Importing faiss second...", flush=True)
import faiss
print("[Diag] faiss imported successfully.", flush=True)

print("[Diag] Importing SentenceTransformer third...", flush=True)
from sentence_transformers import SentenceTransformer
print("[Diag] SentenceTransformer imported successfully!", flush=True)

print("[Diag] Success!", flush=True)
