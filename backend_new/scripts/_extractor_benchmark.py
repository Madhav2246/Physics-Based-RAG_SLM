"""
_extractor_benchmark.py — Side-by-side comparison of Regex vs Two-Stage
(SLM + Regex fallback) value extraction.

Run from the backend/ directory:
    python scripts/_extractor_benchmark.py

Outputs a table showing which extractor correctly identified each value,
and a final score. This table is publishable evidence of the NL extractor
contribution.
"""
from __future__ import annotations
import sys, io, json, math
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics.exploration_engine import ExplorationEngine
from physics.equation_validator  import EquationValidator

# ── Setup engine + SLM (same as the pipeline does it) ───────────────────────
validator = EquationValidator()
engine    = ExplorationEngine(validator.symbols)

# Load the SLM and wire it in — reuses the same weights the pipeline uses
print("Loading SLM for two-stage extraction test...")
try:
    from reasoning.slm_model import TinySLM
    _slm = TinySLM()
    engine.set_slm_model(_slm)
    print("SLM loaded and wired into engine.\n")
except Exception as exc:
    print(f"[WARN] Could not load SLM ({exc}) — benchmark will show regex-only.\n")


# ── 15 benchmark queries ──────────────────────────────────────────────────────
# Each entry: (query, expected {sym: SI_value})
TEST_CASES: list[tuple[str, dict]] = [
    # 1. Standard structured format (regex baseline)
    ("Id=1mA, Vov=0.3V",                                        {"Id": 1e-3, "Vov": 0.3}),
    # 2. Natural language: "drain current is X milliamp"
    ("Assume the drain current is 1 milliamp",                  {"Id": 1e-3}),
    # 3. Natural language: "half a volt"
    ("overdrive voltage is half a volt",                        {"Vov": 0.5}),
    # 4. Mixed voltage with millivolts
    ("Vgs of 1.2 volts, threshold at 400 millivolts",          {"Vgs": 1.2, "Vth": 0.4}),
    # 5. "microamps" spelled out
    ("current of roughly 500 microamps",                        {"Id": 5e-4}),
    # 6. Geometry dimensions
    ("W is 2 microns, L is 100 nanometers",                    {"W": 2e-6, "L": 1e-7}),
    # 7. Scientific notation
    ("Id = 2e-3 A",                                             {"Id": 2e-3}),
    # 8. Both values in words
    ("drain current one milliamp overdrive zero point three",   {"Id": 1e-3, "Vov": 0.3}),
    # 9. tox
    ("tox equals 2nm",                                          {"tox": 2e-9}),
    # 10. Sweep range — NOT a value assignment, should extract nothing
    ("Vov from 0.1V to 0.6V",                                  {}),
    # 11. Pure question — no values stated
    ("what is the W/L ratio",                                   {}),
    # 12. Mixed: node keyword present, should not affect extraction
    ("Id=1mA, Vov=0.3V, using the 5nm node",                   {"Id": 1e-3, "Vov": 0.3}),
    # 13. Gate geometry spelled out
    ("gate width of 3 microns and gate length of 50 nanometers", {"W": 3e-6, "L": 5e-8}),
    # 14. Vsb = 0
    ("Vsb is 0 volts",                                          {"Vsb": 0.0}),
    # 15. Body effect params
    ("gamma equals 0.4, Phi_f is 0.35",                        {"gamma": 0.4, "Phi_f": 0.35}),
]

_TOL = 0.02   # 2% tolerance for float comparison

def _close(a: float, b: float) -> bool:
    if a == b == 0.0:
        return True
    return abs(a - b) / (abs(b) + 1e-30) < _TOL


def _score_extraction(extracted: dict, expected: dict) -> tuple[int, int]:
    """Returns (hits, total_expected).  Hits = expected keys found with correct value."""
    if not expected:
        # Edge case: nothing should be extracted
        hits = 1 if not extracted else 0
        return hits, 1
    hits = sum(
        1 for sym, val in expected.items()
        if sym in extracted and _close(extracted[sym], val)
    )
    return hits, len(expected)


def _run_regex(engine: ExplorationEngine, query: str) -> dict:
    tracker = engine.extract_user_values(query)
    return {k: v.value for k, v in tracker._values.items() if v.provenance == "user"}


def main():
    # engine and SLM are loaded at module level above

    # Check if two-stage extraction is available
    has_slm_extractor = hasattr(engine, "_slm_extractor") and engine._slm_extractor is not None

    print("=" * 80)
    print("EXTRACTOR BENCHMARK — Regex vs Two-Stage (SLM + Regex fallback)")
    print(f"Two-stage extractor available: {has_slm_extractor}")
    print("=" * 80)

    hdr = f"{'#':>2}  {'Query (truncated)':<46}  {'Regex':^7}  {'Expected':<28}"
    if has_slm_extractor:
        hdr += f"  {'TwoStage':^8}"
    print(hdr)
    print("-" * 80)

    regex_total = regex_hits = 0
    ts_total = ts_hits = 0

    for i, (query, expected) in enumerate(TEST_CASES, 1):
        short = query[:44] + ".." if len(query) > 44 else query

        # Regex extraction
        regex_out = _run_regex(engine, query)
        rh, rt = _score_extraction(regex_out, expected)
        regex_hits  += rh
        regex_total += rt
        r_sym = "PASS" if rh == rt else f"{rh}/{rt}"

        exp_str = ", ".join(f"{k}={v:.3g}" for k, v in expected.items()) or "(none)"

        row = f"{i:>2}. {short:<46}  {r_sym:^7}  {exp_str:<28}"

        # Two-stage extraction (only if SLM extractor wired in)
        if has_slm_extractor:
            ts_out = _run_two_stage(engine, query)
            th, tt = _score_extraction(ts_out, expected)
            ts_hits  += th
            ts_total += tt
            t_sym = "PASS" if th == tt else f"{th}/{tt}"
            row += f"  {t_sym:^8}"

        print(row)

    print("-" * 80)
    print(f"\nREGEX       score: {regex_hits}/{regex_total}")
    if has_slm_extractor:
        print(f"TWO-STAGE   score: {ts_hits}/{ts_total}")
    print()
    if not has_slm_extractor:
        print("[NOTE] SLM extractor not yet wired in. This table shows the regex")
        print("       baseline. Re-run after Feature 3 is complete to see improvement.")


def _run_two_stage(engine, query):
    """Call the two-stage extractor if available."""
    try:
        tracker = engine.extract_user_values_twostage(query)
        return {k: v.value for k, v in tracker._values.items() if v.provenance == "user"}
    except Exception as exc:
        print(f"  [WARN] two-stage failed: {exc}")
        return {}


if __name__ == "__main__":
    main()
