import sys

print("[Diag] Importing sentence_transformers...", flush=True)
import sentence_transformers
print("[Diag] sentence_transformers imported successfully.", flush=True)

print("[Diag] Importing huggingface_hub...", flush=True)
import huggingface_hub
print("[Diag] huggingface_hub imported successfully.", flush=True)

print("[Diag] Importing transformers...", flush=True)
import transformers
print("[Diag] transformers imported successfully.", flush=True)

print("[Diag] Importing transformers.models...", flush=True)
import transformers.models
print("[Diag] transformers.models imported successfully.", flush=True)

print("[Diag] Importing sentence_transformers.models...", flush=True)
import sentence_transformers.models
print("[Diag] sentence_transformers.models imported successfully.", flush=True)

print("[Diag] Importing sentence_transformers.util...", flush=True)
import sentence_transformers.util
print("[Diag] sentence_transformers.util imported successfully.", flush=True)

print("[Diag] Importing SentenceTransformer class...", flush=True)
from sentence_transformers import SentenceTransformer
print("[Diag] SentenceTransformer class imported successfully!", flush=True)
