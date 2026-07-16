"""
End-to-end integration test — all three features through RAGPipeline.answer().

This is the one gap the audit flagged: every feature is unit-tested in isolation,
but nothing exercised the real entry point where all three collide. This does.

Run from backend/:
    python -X utf8 scripts/_e2e_integration_test.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.rag_pipeline import RAGPipeline

print("Loading pipeline (model + indexes)...")
pipeline = RAGPipeline()
pipeline.retriever.dense.load_index()
pipeline.retriever.sparse.build_index_from_docs(pipeline.retriever.dense.documents)
print("Ready.\n")

passed = 0
failed = 0

def check(label: str, cond: bool, detail: str = ""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}  {detail}")

# ── Case 1: SWEEP + node profile (all three features in one query) ────────────
print("Case 1: SWEEP + 5nm node — 'plot W/L vs Vov from 0.1 to 0.6 for Id=1mA'")
q1 = "Using 5nm GAA, plot W/L versus Vov from 0.1V to 0.6V for Id=1mA"
r1 = pipeline.answer(q1)
check("mode == SWEEP", r1.get("mode") == "SWEEP", f"got {r1.get('mode')}")
check("node_profile == 5nm_GAA", r1.get("node_profile") == "5nm_GAA",
      f"got {r1.get('node_profile')}")
plot_path = r1.get("sweep_plot_path")
check("sweep_plot_path present", bool(plot_path), f"got {plot_path}")
check("plot file exists on disk",
      bool(plot_path) and Path(plot_path).exists(),
      f"path={plot_path}")
print()

# ── Case 2: EXPLORE single-point, default node ────────────────────────────────
print("Case 2: EXPLORE — 'How do I choose W/L for Id=1mA, Vov=0.5V?'")
q2 = "How do I choose W/L for Id=1mA, Vov=0.5V?"
r2 = pipeline.answer(q2)
check("mode == EXPLORE", r2.get("mode") == "EXPLORE", f"got {r2.get('mode')}")
check("node defaults to 100nm",
      "100nm" in str(r2.get("node_profile")), f"got {r2.get('node_profile')}")
er2 = r2.get("explore_result")
wl = er2.get("numeric") if er2 else None
check("W/L numeric ~8.0 (Id=1mA,Vov=0.5)",
      wl is not None and abs(wl - 8.0) < 0.01, f"got {wl}")
print()

# ── Case 3: LOOKUP — pure definition, no design intent ────────────────────────
print("Case 3: LOOKUP — 'What is the drain current equation in saturation?'")
q3 = "What is the drain current equation in saturation?"
r3 = pipeline.answer(q3)
check("mode == LOOKUP", r3.get("mode") == "LOOKUP", f"got {r3.get('mode')}")
check("explore_result is None in LOOKUP", r3.get("explore_result") is None)
print()

# ── Case 4: SWEEP with bad target collision falls back to EXPLORE ──────────────
print("Case 4: SWEEP fallback — sweeping the solved-for variable")
q4 = "plot Id versus Id from 0.1 to 0.6"   # nonsensical — must not crash
r4 = pipeline.answer(q4)
check("no crash, returns a mode", r4.get("mode") in ("SWEEP", "EXPLORE", "LOOKUP"),
      f"got {r4.get('mode')}")
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  E2E INTEGRATION: {passed} passed, {failed} failed")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
