"""
End-to-end explore mode demo — through the full pipeline.
Tests both EXPLORE and LOOKUP paths in the same session.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.retriever.dense.load_index()
pipeline.retriever.sparse.build_index_from_docs(pipeline.retriever.dense.documents)

TESTS = [
    # EXPLORE — user supplies all numeric values
    ("How do I choose W/L for Id = 1mA, Vov = 0.5V?",   "EXPLORE", "all user"),
    # EXPLORE — user supplies some values (defaults used for mu, Cox)
    ("How do I choose W/L for a drain current of Id = 500uA?", "EXPLORE", "partial user"),
    # LOOKUP — should hit corpus equation
    ("What is the MOSFET drain current equation in saturation?", "LOOKUP", "lookup"),
    # EXPLORE — target not in retrieved equation (expect honest failure)
    ("How do I choose tox for a target Cox = 0.01 F/m2?", "EXPLORE", "tox target"),
]

for query, expected_mode, label in TESTS:
    print("=" * 70)
    print(f"[{label}] {query}")
    print("-" * 70)

    result = pipeline.answer(query)
    mode   = result.get("mode", "?")
    eq_line = result["response"].split("\n")[0]

    print(f"  Mode       : {mode}  (expected {expected_mode})")
    print(f"  Equation   : {eq_line}")
    print(f"  Symbolic   : {result['symbolic_validation']}")
    print(f"  Dimensional: {result['dimension_validation']}")
    print(f"  Confidence : {result['confidence_score']:.3f} -> {result['confidence_label']}")

    if result.get("explore_result"):
        er = result["explore_result"]
        t  = er.get("tracker")
        print(f"  Provenance : {t.provenance_fraction:.0%} user-supplied" if t else "")
        print(f"  Sanity     : {'PASS' if er.get('sanity_ok') else 'FAIL/not run'}")
    print()

print("=" * 70)
print("CONFIDENCE WEIGHT CHECK (all-defaults explore case):")
from utils.confidence_engine import ConfidenceEngine
ce = ConfidenceEngine()
score_full, _ = ce.score_explore(True, True, True, True, True, 1.0)
score_def,  _ = ce.score_explore(True, True, True, True, True, 0.0)
score_half, _ = ce.score_explore(True, True, True, True, True, 0.5)
score_none, _ = ce.score_explore(False, False, False, False, False, 0.0)
print(f"  All user    = {score_full:.3f}  (target: ~0.90)")
print(f"  All defaults= {score_def:.3f}  (target: ~0.65)")
print(f"  Half/half   = {score_half:.3f}  (target: ~0.775)")
print(f"  No solution = {score_none:.3f}  (target: LOW)")
print("=" * 70)
