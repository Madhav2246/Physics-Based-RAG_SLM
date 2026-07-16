"""
build_gold_anchors.py
---------------------
Build a chunking-independent ground truth for Stage-3 retrieval.

The golden QA's `source_chunk` is a 301-char prefix of a raw-corpus PARAGRAPH
(synthesize_data's old blank-line split). That paragraph is the TRUE answer-bearing
span — it exists regardless of how we later chunk the index. We anchor ground truth
to that paragraph, so we can sweep chunk size / overlap / embedder freely and still
ask "did retrieval surface the chunk(s) that contain the gold paragraph?".

Output: data/evaluation/gold_anchors.json
  [{ "id", "difficulty", "question", "anchor_text" }, ...]
anchor_text = the full raw paragraph (clean, untruncated, the real source).

Run from backend_new/:
  python scripts/build_gold_anchors.py
"""
import io, json, re, sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT   = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "evaluation" / "nvidia_golden_qa.jsonl"
CORPUS = ROOT / "data" / "corpus"
OUT    = ROOT / "data" / "evaluation" / "gold_anchors.json"


def _norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def main():
    golden = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]

    paras = []
    for f in sorted(CORPUS.glob("*.txt")):
        t = f.read_text(encoding="utf-8", errors="replace")
        paras.extend([p.strip() for p in re.split(r"\n{2,}", t) if len(p.split()) > 40])
    nparas = [_norm(p) for p in paras]
    print(f"{len(golden)} QA, {len(paras)} raw paragraphs.")

    anchors, miss = [], 0
    for g in golden:
        stub = g["source_chunk"].rstrip("…").rstrip("�").rstrip()
        core = _norm(stub)[20:160]
        cand = [i for i, p in enumerate(nparas) if core and core in p]
        if not cand:
            # fallback: longest common probe shrink
            for L in (120, 90, 60):
                core2 = _norm(stub)[20:20 + L]
                cand = [i for i, p in enumerate(nparas) if core2 and core2 in p]
                if cand:
                    break
        if not cand:
            miss += 1
            anchor = stub          # last resort: the stub itself
        else:
            # shortest matching paragraph = tightest true span
            anchor = paras[min(cand, key=lambda i: len(paras[i]))]
        anchors.append({
            "id": g["id"], "difficulty": g["difficulty"],
            "question": g["question"], "anchor_text": anchor,
        })

    OUT.write_text(json.dumps(anchors, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(anchors)} anchors ({miss} fell back to stub) -> {OUT}")


if __name__ == "__main__":
    main()
