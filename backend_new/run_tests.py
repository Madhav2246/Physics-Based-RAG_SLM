# -- Windows UTF-8 / environment setup ----------------------------------------
import sys
import io

# Force stdout/stderr to UTF-8 so Greek letters (μ, γ, Φ) print without crashing
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pathlib
_orig_read_text = pathlib.Path.read_text
def _utf8_read_text(self, encoding=None, errors=None):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors)
pathlib.Path.read_text = _utf8_read_text

import builtins
_orig_open = builtins.open
def _utf8_open(*args, **kwargs):
    mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
    if "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _orig_open(*args, **kwargs)
builtins.open = _utf8_open

import os
os.environ["HF_HOME"] = "d:/S6/NLP/Physics_Based_RAG_SLM/hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

# -----------------------------------------------------------------------------
import traceback
from pipeline.rag_pipeline import RAGPipeline

# Standard 5-query test suite
TEST_QUERIES = [
    "What is the MOSFET drain current equation in saturation?",
    "How does temperature affect threshold voltage?",
    "When does a MOSFET operate in saturation?",
    "What is the equation for the body effect?",
    "What is the subthreshold swing equation?",
]

# Minimal corpus — ASCII-safe equivalents so corpus itself doesn't cause codec errors
TEST_CORPUS = [
    "Drain current equation in saturation: Id = 0.5 * mu * Cox * (W/L) * (Vgs - Vth)^2",
    "MOSFET operates in saturation when Vds > Vgs - Vth.",
    "Threshold voltage decreases with increasing temperature due to reduced bandgap.",
    "Body effect: Vth = Vth0 + gamma*(sqrt(2*Phi_f + Vsb) - sqrt(2*Phi_f))",
    "Subthreshold swing: SS = (kT/q) * ln(10) * (1 + Cd/Cox)",
]


def _safe(text: str, limit: int = None) -> str:
    """Replace unencodable chars with ? for safe console printing."""
    s = text.encode("utf-8", errors="replace").decode("utf-8")
    return s[:limit] + "..." if limit and len(s) > limit else s


def run_tests(output_file: str = "test_output_results.txt") -> None:
    print("=" * 60)
    print("Physics-Based RAG SLM - Test Suite (5 queries)")
    print("=" * 60)

    try:
        pipeline = RAGPipeline()
        pipeline.build(TEST_CORPUS)
        print("[test] Pipeline ready. Running queries...\n")
    except Exception:
        print("FATAL: Pipeline failed to initialise:")
        traceback.print_exc()
        return

    results = []
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"--- Test {i}: {query} ---")
        try:
            result = pipeline.answer(query)
            results.append((i, query, result))

            print(f"  Evidence       : {_safe(str(result['evidence']), 200)}")
            print(f"  Response       : {_safe(result['response'], 200)}")
            print(f"  Symbolic       : {_safe(result['symbolic_validation'])}")
            print(f"  Dimension      : {_safe(result['dimension_validation'])}")
            print(f"  Numerical      : {_safe(result['numerical_validation'])}")
            if result.get("semantic_similarity") is not None:
                print(f"  Semantic Sim.  : {result['semantic_similarity']:.3f}")
            print(f"  Confidence     : {result['confidence_score']} - {result['confidence_label']}")
            print(f"  Stability      : {result['uncertainty_score']} - {result['stability_label']}")
        except Exception:
            print(f"  ERROR in test {i}:")
            traceback.print_exc()
        print()

    # Write full results to UTF-8 file (no codec issues here)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, query, r in results:
            f.write(f"=== Test {i} ===\n")
            f.write(f"Query: {query}\n")
            f.write(f"Retrieved Evidence: {r['evidence']}\n")
            f.write(f"Response: {r['response']}\n")
            f.write(f"Symbolic Validation: {r['symbolic_validation']}\n")
            f.write(f"Dimension Validation: {r['dimension_validation']}\n")
            f.write(f"Numerical Validation: {r['numerical_validation']}\n")
            if r.get("semantic_similarity") is not None:
                f.write(f"Semantic Similarity: {r['semantic_similarity']:.4f}\n")
            f.write(f"Confidence Score: {r['confidence_score']}\n")
            f.write(f"Confidence Level: {r['confidence_label']}\n")
            f.write(f"Uncertainty Score: {r['uncertainty_score']}\n")
            f.write(f"Stability Label: {r['stability_label']}\n")
            f.write("\n" + "=" * 40 + "\n\n")

    print(f"[test] Full results written to '{output_file}'")


if __name__ == "__main__":
    run_tests()
