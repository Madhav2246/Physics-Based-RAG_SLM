# -- Windows UTF-8 workaround -------------------------------------------------
# Forces UTF-8 for all file I/O on Windows where the default is cp1252.
# Alternative: set PYTHONUTF8=1 env var or use `python -X utf8 main.py`
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

# -- Environment setup ---------------------------------------------------------
import os
os.environ["HF_HOME"] = "d:/S6/NLP/Physics_Based_RAG_SLM/hf_cache"
# Remove broken corporate SSL cert if the file no longer exists
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

import traceback
import glob
import utils.config as cfg
from pipeline.rag_pipeline import RAGPipeline

# -----------------------------------------------------------------------------

def load_corpus(corpus_dir: str = cfg.CORPUS_DIR, max_docs: int = 2000) -> list[str]:
    """
    Load paragraph-level chunks from all .txt files in corpus_dir.
    Falls back to a small toy corpus if no corpus files are found.
    """
    txt_files = glob.glob(os.path.join(corpus_dir, "*.txt"))

    if not txt_files:
        print(f"[main] No corpus files found in '{corpus_dir}'. Using toy corpus.")
        return [
            "Drain current equation in saturation: Id = 0.5 * μ * Cox * (W/L) * (Vgs - Vth)^2",
            "MOSFET operates in saturation when Vds > Vgs - Vth.",
            "Threshold voltage decreases linearly with temperature.",
            "The body effect: Vth = Vth0 + γ(sqrt(2Φf + Vsb) - sqrt(2Φf))",
            "Subthreshold swing SS = (kT/q) * ln(10) * (1 + Cd/Cox)",
        ]

    docs = []
    for path in sorted(txt_files):
        try:
            text = open(path, encoding="utf-8").read()
            # Split into paragraphs; skip very short ones
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) > 60:
                    docs.append(para)
                    if len(docs) >= max_docs:
                        break
        except Exception as e:
            print(f"[main] Warning: could not read {path}: {e}")
        if len(docs) >= max_docs:
            break

    print(f"[main] Loaded {len(docs)} document chunks from {len(txt_files)} corpus files.")
    return docs


def print_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print(f"ANSWER:\n{result['response']}")
    print(f"\nEvidence used: {result['evidence']}")
    print(f"\nSymbolic  : {result['symbolic_validation']}")
    print(f"Dimension : {result['dimension_validation']}")
    print(f"Numerical : {result['numerical_validation']}")
    if result.get("semantic_similarity") is not None:
        print(f"Sem. Sim. : {result['semantic_similarity']:.3f}")
    print(f"\nConfidence: {result['confidence_score']} — {result['confidence_label']}")
    print(f"Stability : {result['uncertainty_score']} — {result['stability_label']}")
    print("=" * 60)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("Physics-Based RAG SLM — Demo")
        print("=" * 60)

        # Load documents
        docs = load_corpus()

        # Build pipeline & index
        pipeline = RAGPipeline()
        print("\n[main] Building retrieval index...")
        pipeline.build(docs)
        print("[main] Index built. Ready.\n")

        # Run demo queries
        queries = [
            "What is the MOSFET drain current equation in saturation?",
            "How does temperature affect threshold voltage?",
            "What is the equation for the body effect?",
        ]

        for query in queries:
            print(f"\nQUERY: {query}")
            result = pipeline.answer(query)
            print_result(result)

    except Exception:
        print("\nCRITICAL ERROR:")
        traceback.print_exc()