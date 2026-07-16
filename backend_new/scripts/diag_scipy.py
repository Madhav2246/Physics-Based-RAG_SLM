import sys

print("[Diag] Importing scipy...", flush=True)
import scipy
print("[Diag] scipy imported successfully.", flush=True)

print("[Diag] Importing scipy.stats (this imports C extensions)...", flush=True)
import scipy.stats
print("[Diag] scipy.stats imported successfully.", flush=True)

print("[Diag] Importing scipy.interpolate...", flush=True)
import scipy.interpolate
print("[Diag] scipy.interpolate imported successfully.", flush=True)

print("[Diag] Importing scipy.interpolate._rbfinterp_xp...", flush=True)
from scipy.interpolate import _rbfinterp_xp
print("[Diag] scipy.interpolate._rbfinterp_xp imported successfully.", flush=True)

print("[Diag] All scipy diagnostics completed successfully!", flush=True)
