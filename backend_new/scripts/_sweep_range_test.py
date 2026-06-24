"""
Standalone range-parser test — must pass 100% before SweepEngine plot code runs.
This is the same discipline that caught the negative-Vth bug: prove the brittle
regex parser in isolation first.

Run from backend/:
    python -X utf8 scripts/_sweep_range_test.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics.sweep_engine import parse_sweep_request

# (query, expected_sweep_var, expected_start, expected_stop)  — None means "no sweep"
CASES = [
    ("Plot W/L versus Vov from 0.1V to 0.6V for Id=1mA", "Vov", 0.1, 0.6),
    ("sweep Id from 1uA to 10mA",                         "Id",  1e-6, 1e-2),
    ("what W/L for Id=1mA, Vov=0.3V",                     None,  None, None),
    ("Vov vs W/L, range 0.2 to 0.8",                      "Vov", 0.2, 0.8),
    # extra robustness cases
    ("Plot gm versus Vov from 0.2V to 0.5V",              "Vov", 0.2, 0.5),
    ("sweep tox from 1nm to 5nm",                         "tox", 1e-9, 5e-9),
    ("Define threshold voltage",                          None,  None, None),
]

def approx(a, b, tol=1e-9):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(abs(b), 1.0)

print("=" * 78)
print("SWEEP RANGE PARSER TEST")
print("=" * 78)

all_pass = True
for query, exp_var, exp_start, exp_stop in CASES:
    req = parse_sweep_request(query)

    if exp_var is None:
        ok = req is None
        got = "None" if req is None else f"{req.sweep_var}[{req.start},{req.stop}]"
    else:
        ok = (req is not None
              and req.sweep_var == exp_var
              and approx(req.start, exp_start)
              and approx(req.stop, exp_stop))
        got = (f"{req.sweep_var}[{req.start:.4g},{req.stop:.4g}]"
               if req is not None else "None")

    all_pass = all_pass and ok
    status = "PASS" if ok else "FAIL"
    exp_str = "None" if exp_var is None else f"{exp_var}[{exp_start:.4g},{exp_stop:.4g}]"
    print(f"  [{status}] '{query[:50]:<50}' -> {got:<22} (expected {exp_str})")

print("-" * 78)
print("ALL PASS" if all_pass else "SOME FAILED — do not build plot code until green")
sys.exit(0 if all_pass else 1)
