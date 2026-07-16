import sys

print("[Diag] Importing numpy...", flush=True)
import numpy as np

print("[Diag] Importing sklearn...", flush=True)
import sklearn
print("[Diag] sklearn imported successfully.", flush=True)

print("[Diag] Importing pairwise_distances from sklearn.metrics...", flush=True)
from sklearn.metrics import pairwise_distances
print("[Diag] pairwise_distances imported successfully.", flush=True)

print("[Diag] Importing clone from sklearn.base...", flush=True)
from sklearn.base import clone
print("[Diag] clone imported successfully.", flush=True)

print("[Diag] Success!", flush=True)
