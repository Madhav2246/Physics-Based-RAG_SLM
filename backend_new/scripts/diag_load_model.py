import os, sys
os.environ["HF_HOME"] = "d:/S6/NLP/Physics_Based_RAG_SLM/hf_cache"

print("[Diag] Importing sentence_transformers...", flush=True)
from sentence_transformers import SentenceTransformer

print("[Diag] Initialising SentenceTransformer('BAAI/bge-large-en-v1.5')...", flush=True)
try:
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    print("[Diag] Model loaded successfully!", flush=True)
except Exception as e:
    import traceback
    print("[Diag] Model loading failed:", flush=True)
    traceback.print_exc()
