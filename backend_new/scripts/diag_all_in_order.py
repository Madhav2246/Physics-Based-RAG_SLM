import sys

print("[Diag] Importing numpy...", flush=True)
import numpy as np

print("[Diag] Importing scipy...", flush=True)
import scipy

print("[Diag] Importing sympy...", flush=True)
import sympy

print("[Diag] Importing torch...", flush=True)
import torch

print("[Diag] Importing faiss...", flush=True)
import faiss

print("[Diag] Importing sentence_transformers...", flush=True)
import sentence_transformers

print("[Diag] Importing SentenceTransformer class...", flush=True)
from sentence_transformers import SentenceTransformer

print("[Diag] Importing openai...", flush=True)
import openai

print("[Diag] All core imports completed successfully in exact order!", flush=True)
