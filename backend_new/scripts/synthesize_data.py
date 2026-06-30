import argparse
import json
import os
import re
import sys
import time
import random
import hashlib
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# -- project root --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# -- output path ---------------------------------------------------------------
OUTPUT_DIR    = PROJECT_ROOT / "data" / "evaluation"
OUTPUT_FILE   = OUTPUT_DIR / "nvidia_golden_qa.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "live.json"

# -- NVIDIA NIM config ---------------------------------------------------------
NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY", "")
if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY environment variable not set")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL           = "meta/llama-3.1-70b-instruct"

# -- dataset targets -----------------------------------------------------------
TARGETS = {
    "easy":   40,
    "medium": 40,
    "hard":   20,
}
TOTAL_TARGET = sum(TARGETS.values())   # 100

# -- difficulty definitions ----------------------------------------------------
DIFFICULTY_CONFIG = {
    "easy": {
        "label": "EASY",
        "description": (
            "A single-equation recall question. The student should be able to "
            "answer by citing one standard MOSFET equation directly (e.g., "
            "drain current in linear or saturation, threshold voltage definition, "
            "transconductance). The answer must contain the full equation with "
            "all symbols defined."
        ),
        "example_q": "What is the drain current equation for a MOSFET in saturation?",
        "example_a": "Id = 0.5 * mu_n * Cox * (W/L) * (Vgs - Vth)^2, where mu_n is electron mobility, Cox is gate oxide capacitance per unit area, W/L is the aspect ratio, and Vth is the threshold voltage.",
    },
    "medium": {
        "label": "MEDIUM",
        "description": (
            "A multi-step reasoning question requiring the student to identify "
            "which regime a device is in, derive a relationship between two "
            "parameters, or explain the physical origin of an effect (e.g., "
            "body effect, channel length modulation, velocity saturation). "
            "The answer must include at least one equation AND a one-sentence "
            "physical interpretation."
        ),
        "example_q": "How does substrate bias (Vsb) affect the threshold voltage in a MOSFET, and what is the governing equation?",
        "example_a": "Substrate bias increases threshold voltage via the body effect: Vth = Vth0 + gamma * (sqrt(2*Phi_f + Vsb) - sqrt(2*Phi_f)), where gamma is the body effect coefficient and Phi_f is the Fermi potential. Physically, the reverse-biased body-source junction widens the depletion region, requiring more gate charge to invert the channel.",
    },
    "hard": {
        "label": "HARD",
        "description": (
            "An advanced question requiring synthesis of multiple equations, "
            "small-signal model analysis, second-order effects (DIBL, subthreshold "
            "swing degradation, gate tunneling), or derivation from first principles. "
            "The answer must contain ≥2 distinct equations and demonstrate "
            "understanding of the limiting physics."
        ),
        "example_q": "Derive the subthreshold swing and explain why 60 mV/decade is the fundamental limit at room temperature.",
        "example_a": "SS = (kT/q) * ln(10) * (1 + Cd/Cox) mV/decade. In the ideal case Cd<<Cox, giving SS_min = (kT/q)*ln(10) ≈ 60 mV/decade at 300K. Here k is Boltzmann's constant, T is temperature, q is electron charge, Cd is depletion capacitance, and Cox is oxide capacitance. The limit arises because subthreshold current is thermally activated (Boltzmann statistics) — the exponential tail of the carrier distribution sets a floor on how sharply the device can switch.",
    },
}

# -- deduplication similarity threshold ---------------------------------------
DEDUP_THRESHOLD = 0.82   # cosine sim above this → treat as duplicate


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_openai_client():
    """Lazy import of openai (compatible with NVIDIA NIM endpoint)."""
    try:
        from openai import OpenAI
        return OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    except ImportError:
        raise ImportError("openai package not installed — run: pip install openai")


def _load_embed_model():
    """Load MiniLM for deduplication similarity checks."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        raise ImportError("sentence-transformers not installed")


def _cosine_sim(a, b) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _load_corpus_chunks(corpus_dir: Path) -> List[Tuple[int, str]]:
    """Load the ACTUAL retrieval-index chunks as (chunk_id, text) pairs.

    The QA must be generated from the SAME chunks the retriever indexes, otherwise
    the ground-truth chunk cannot appear in the index and Hit@k is meaningless.
    (Earlier this split corpus .txt by blank-line paragraphs — a different chunk
    space than the 512-word sentence windows in docs.json — which broke Stage 3.)

    Falls back to paragraph-splitting the corpus only if docs.json is absent.
    """
    docs_path = PROJECT_ROOT / "data" / "embeddings" / "docs.json"
    if docs_path.exists():
        docs = json.loads(docs_path.read_text(encoding="utf-8"))
        return list(enumerate(docs))   # (chunk_id, text)

    print("[WARN] docs.json not found — falling back to paragraph split "
          "(NOTE: resulting chunk_ids will NOT match the retrieval index).")
    chunks = []
    cid = 0
    for txt_file in sorted(corpus_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        for p in re.split(r'\n{2,}', text):
            p = p.strip()
            if len(p.split()) > 40:
                chunks.append((cid, p))
                cid += 1
    return chunks


def _quality_filter(q: str, a: str) -> Tuple[bool, str]:
    """
    Returns (passes, reason).
    Reject if:
      - Question < 15 chars
      - Answer < 40 chars
      - Answer contains no equation symbols
      - Question is a duplicate of an existing question (caller handles this)
    """
    if len(q.strip()) < 15:
        return False, "question too short"
    if len(a.strip()) < 40:
        return False, "answer too short"
    eq_symbols = re.findall(r'[=+\-*/^_\\]', a)
    if len(eq_symbols) < 2:
        return False, f"answer has no equation (found {len(eq_symbols)} math symbols)"
    return True, "ok"


def _load_existing_entries() -> List[Dict]:
    """Load already-generated entries from the output JSONL (for resume)."""
    if not OUTPUT_FILE.exists():
        return []
    entries = []
    for line in OUTPUT_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def _save_entry(entry: Dict):
    """Append one entry to the JSONL file (live write)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _update_progress(counts: Dict, total: int, current_diff: str, status: str,
                     last_question: str = "", last_id: str = ""):
    by_diff = {}
    for diff, target in TARGETS.items():
        done = counts.get(diff, 0)
        by_diff[diff] = {
            "done":      done,
            "target":    target,
            "remaining": max(0, target - done),
            "complete":  done >= target,
        }

    payload = {
        "status":        status,
        "updated_at":    datetime.datetime.utcnow().isoformat() + "Z",
        "overall": {
            "done":    total,
            "target":  TOTAL_TARGET,
            "percent": round(total / TOTAL_TARGET * 100, 1),
        },
        "by_difficulty": by_diff,
        "last": {
            "id":         last_id,
            "difficulty": current_diff,
            "question":   last_question,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert semiconductor device physics professor creating exam questions.
Your task: given a passage from a physics textbook, generate ONE question-answer pair at the specified difficulty level.

STRICT RULES:
1. The question MUST be answerable using ONLY information in the provided passage.
2. The answer MUST contain at least one complete physics equation with all symbols defined.
3. Do NOT repeat questions that are semantically similar to the provided examples.
4. Output ONLY valid JSON — no preamble, no explanation, no markdown fences.
5. JSON format: {"question": "...", "answer": "...", "equation": "the key equation as a string", "symbols": {"symbol": "definition", ...}}"""


def _build_user_prompt(chunk: str, difficulty: str, existing_questions: List[str]) -> str:
    cfg = DIFFICULTY_CONFIG[difficulty]
    existing_sample = "\n".join(f"  - {q}" for q in existing_questions[-8:]) if existing_questions else "  (none yet)"

    return f"""DIFFICULTY: {cfg['label']}
DIFFICULTY DESCRIPTION: {cfg['description']}

EXAMPLE OF THIS DIFFICULTY LEVEL:
  Q: {cfg['example_q']}
  A: {cfg['example_a']}

RECENT QUESTIONS ALREADY GENERATED (DO NOT REPEAT THESE TOPICS):
{existing_sample}

PHYSICS PASSAGE TO USE:
\"\"\"
{chunk[:1200]}
\"\"\"

Generate ONE {cfg['label']}-difficulty question-answer pair from this passage.
Output ONLY the JSON object, nothing else."""


# -----------------------------------------------------------------------------
# Generation loop
# -----------------------------------------------------------------------------

def generate_dataset(corpus_dir: Path, reset: bool = False):
    if reset and OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print("[OK] Reset: cleared existing output file.")

    client      = _load_openai_client()
    embed_model = _load_embed_model()

    print(f"📚  Loading corpus from {corpus_dir}…")
    chunks = _load_corpus_chunks(corpus_dir)
    if not chunks:
        print(f"[X]  No corpus chunks found in {corpus_dir}. Run extract_pdfs.py first.")
        sys.exit(1)
    print(f"    {len(chunks)} paragraphs loaded.\n")

    existing      = _load_existing_entries()
    counts        = {"easy": 0, "medium": 0, "hard": 0}
    existing_qs   = []
    existing_vecs = []

    for entry in existing:
        diff = entry.get("difficulty", "easy")
        if diff in counts:
            counts[diff] += 1
        existing_qs.append(entry["question"])

    if existing:
        existing_vecs = embed_model.encode(existing_qs, show_progress_bar=False).tolist()
        total_done = sum(counts.values())
        print(f"▶  Resuming — {total_done}/{TOTAL_TARGET} already done "
              f"({counts['easy']} easy / {counts['medium']} medium / {counts['hard']} hard)\n")

    todo = []
    for diff in ["easy", "medium", "hard"]:
        remaining = TARGETS[diff] - counts[diff]
        todo.extend([diff] * max(0, remaining))

    if not todo:
        print("✅  Dataset already complete (100/100). Use --reset to regenerate.")
        return

    print(f"🎯  Need to generate: {len([d for d in todo if d=='easy'])} easy  "
          f"/ {len([d for d in todo if d=='medium'])} medium  "
          f"/ {len([d for d in todo if d=='hard'])} hard\n")
    print("-" * 65)

    shuffled_chunks = chunks.copy()
    random.shuffle(shuffled_chunks)
    chunk_idx = 0

    for i, difficulty in enumerate(todo):
        total_done  = sum(counts.values())
        overall_num = total_done + 1

        print(f"\n[{overall_num:>3}/100]  Generating {difficulty.upper()} question…", end=" ", flush=True)
        _update_progress(counts, total_done, difficulty, "generating")

        attempts = 0
        success  = False

        while attempts < 8:
            chunk_id, chunk = shuffled_chunks[chunk_idx % len(shuffled_chunks)]
            chunk_idx += 1
            attempts  += 1

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": _build_user_prompt(
                            chunk, difficulty, existing_qs
                        )},
                    ],
                    temperature=0.7,
                    max_tokens=512,
                )
                raw = response.choices[0].message.content.strip()
            except Exception as exc:
                err_str = str(exc).lower()
                print(f"\n    [WARN]  API error (attempt {attempts}): {exc}")
                
                # Check for API key exhaustion or rate limits
                if "401" in err_str or "quota" in err_str or "exhausted" in err_str or "unauthorized" in err_str:
                    print(f"\n    [FAIL]  API Key exhausted or invalid. Stopping.")
                    _update_progress(counts, sum(counts.values()), difficulty, "need to change api key")
                    sys.exit(1)
                    
                time.sleep(3)
                continue

            try:
                raw_clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
                parsed    = json.loads(raw_clean)
                question  = parsed.get("question", "").strip()
                answer    = parsed.get("answer", "").strip()
                equation  = parsed.get("equation", "").strip()
                symbols   = parsed.get("symbols", {})
            except (json.JSONDecodeError, AttributeError):
                print(f"[X](parse)", end=" ", flush=True)
                continue

            passes, reason = _quality_filter(question, answer)
            if not passes:
                print(f"[X]({reason})", end=" ", flush=True)
                continue

            q_vec = embed_model.encode([question], show_progress_bar=False)[0].tolist()
            is_dup = False
            if existing_vecs:
                import numpy as np
                sims = [_cosine_sim(q_vec, ev) for ev in existing_vecs]
                max_sim = max(sims)
                if max_sim > DEDUP_THRESHOLD:
                    print(f"[X](dup sim={max_sim:.2f})", end=" ", flush=True)
                    is_dup = True

            if is_dup:
                continue

            entry = {
                "id":           f"{difficulty[0].upper()}{counts[difficulty]+1:03d}",
                "difficulty":   difficulty,
                "question":     question,
                "answer":       answer,
                "equation":     equation,
                "symbols":      symbols,
                # chunk_id ties the QA to the exact retrieval-index chunk it came
                # from — this is the Stage-3 Hit@k ground truth. source_chunk stores
                # the FULL chunk (not a 301-char stub) for transparency.
                "chunk_id":     chunk_id,
                "chunk_id_method": "generation",
                "chunk_id_score":  1.0,
                "source_chunk": chunk,
                "source_chunk_full": chunk,
                "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "attempts":     attempts,
            }

            _save_entry(entry)
            existing_qs.append(question)
            existing_vecs.append(q_vec)
            counts[difficulty] += 1
            success = True

            total_done = sum(counts.values())
            _update_progress(counts, total_done, difficulty, "ok",
                             last_question=question, last_id=entry["id"])

            print(f"[OK]  [{counts['easy']}E / {counts['medium']}M / {counts['hard']}H]  "
                  f"Q: {question[:65]}{'…' if len(question)>65 else ''}")
            break

        if not success:
            print(f"\n    [WARN]  Skipped after {attempts} failed attempts — moving on.")
            _update_progress(counts, sum(counts.values()), difficulty, "skipped")

    total_done = sum(counts.values())
    _update_progress(counts, total_done, "-", "complete")
    print("\n" + "=" * 65)
    print(f"✅  DONE  —  {total_done} questions written to {OUTPUT_FILE}")
    print(f"    Easy:   {counts['easy']}/40")
    print(f"    Medium: {counts['medium']}/40")
    print(f"    Hard:   {counts['hard']}/20")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize golden Q&A dataset via Llama-3-70B.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus",
        help="Path to the corpus directory (default: data/corpus/)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing output and regenerate from scratch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for chunk shuffling (default: 42).",
    )
    args = parser.parse_args()
    random.seed(args.seed)

    generate_dataset(corpus_dir=args.corpus, reset=args.reset)
