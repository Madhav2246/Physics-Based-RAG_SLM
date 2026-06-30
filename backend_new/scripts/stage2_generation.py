"""
stage2_generation.py
--------------------
Stage 2: Generation Quality — BERTScore, ROUGE-L, BLEU-4, Faithfulness,
Answer Relevancy.

Runs entirely from stored texts in answers_dump.jsonl (no GPU / model needed).
Uses new_checker to re-pick best-of-N (same selection as Stage 1).

Metrics:
  BERTScore F1  — semantic overlap vs 70B reference
  ROUGE-L       — longest common subsequence vs 70B reference
  BLEU-4        — n-gram precision vs 70B reference
  Faithfulness  — cosine_sim(answer_embed, corpus_eq_embed)  [SYS vs 70B]
  Ans. Relevancy — cosine_sim(answer_embed, question_embed) [all sides]

IMPORTANT: 70B is BOTH baseline and reference. BERTScore/ROUGE/BLEU of 70B vs
itself = 1.0 by construction. Metrics are reported for SYS and RAW vs 70B
reference — any SYS gap is expected and explained (§7.3 of MD).

Run from backend_new/:
  python scripts/stage2_generation.py
"""
import io, json, sys, time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DUMP    = ROOT / "data" / "evaluation" / "answers_dump.jsonl"
OUT_DIR = ROOT / "data" / "evaluation" / "stage2_generation"
OUT_JSON = OUT_DIR / "stage2_generation.json"

# ---------------------------------------------------------------------------
def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0

def _best_sys(corpus_eq, raw_samples, scorer):
    """Re-pick best-of-N SYS (same logic as stage1_physics_new.py)."""
    texts = [(f"Equation: {corpus_eq}\n\n{s}" if corpus_eq
              else f"Equation: NOT FOUND IN CORPUS\n\n{s}") for s in raw_samples]
    scores = [scorer(t, "SYS")["total"] for t in texts]
    return texts[scores.index(max(scores))]

def _best_raw(raw_samples, scorer):
    scores = [scorer(s, "RAW")["total"] for s in raw_samples]
    return raw_samples[scores.index(max(scores))]

# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("Loading stored answers …")
    records = [json.loads(l) for l in DUMP.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(records)
    print(f"  {n} records loaded from {DUMP.name}")

    # -- Build SYS/RAW texts (re-score to pick best-of-N) -------------------
    print("Picking best-of-N with new_checker …")
    from physics.new_checker import score_text
    sys_texts, raw_texts, b70_texts, questions, diffs = [], [], [], [], []
    corpus_eqs = []
    for r in records:
        sys_texts.append(_best_sys(r["corpus_eq"], r["raw_samples"], score_text))
        raw_texts.append(_best_raw(r["raw_samples"], score_text))
        b70_texts.append(r["b70_text"])
        questions.append(r["question"])
        diffs.append(r.get("difficulty", "easy"))
        corpus_eqs.append(r.get("corpus_eq", "") or "")
    print(f"  done ({time.time()-t0:.0f}s)")

    # -- BERTScore -----------------------------------------------------------
    print("\nComputing BERTScore (roberta-large, CPU) …")
    from bert_score import score as bscore
    # SYS vs 70B ref
    P_sys, R_sys, F_sys = bscore(sys_texts, b70_texts, lang="en", verbose=False)
    F_sys = F_sys.tolist()
    # RAW vs 70B ref
    P_raw, R_raw, F_raw = bscore(raw_texts, b70_texts, lang="en", verbose=False)
    F_raw = F_raw.tolist()
    print(f"  BERTScore done ({time.time()-t0:.0f}s)")
    print(f"  SYS F1 mean={_avg(F_sys):.4f}  RAW F1 mean={_avg(F_raw):.4f}")

    # -- ROUGE-L + BLEU-4 ----------------------------------------------------
    print("\nComputing ROUGE-L + BLEU-4 …")
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    from rouge_score import rouge_scorer as rs_mod
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    smooth = SmoothingFunction().method1
    rouge_eval = rs_mod.RougeScorer(["rougeL"], use_stemmer=False)

    rl_sys, rl_raw, bleu_sys, bleu_raw = [], [], [], []
    for hyp_sys, hyp_raw, ref in zip(sys_texts, raw_texts, b70_texts):
        rl_sys.append(rouge_eval.score(ref, hyp_sys)["rougeL"].fmeasure)
        rl_raw.append(rouge_eval.score(ref, hyp_raw)["rougeL"].fmeasure)
        ref_tok  = word_tokenize(ref.lower())
        hyp_s_tok = word_tokenize(hyp_sys.lower())
        hyp_r_tok = word_tokenize(hyp_raw.lower())
        bleu_sys.append(sentence_bleu([ref_tok], hyp_s_tok,
                                      weights=(0.25,)*4, smoothing_function=smooth))
        bleu_raw.append(sentence_bleu([ref_tok], hyp_r_tok,
                                      weights=(0.25,)*4, smoothing_function=smooth))
    print(f"  ROUGE-L SYS={_avg(rl_sys):.4f} RAW={_avg(rl_raw):.4f}")
    print(f"  BLEU-4  SYS={_avg(bleu_sys):.4f} RAW={_avg(bleu_raw):.4f}")

    # -- Faithfulness (cosine sim: answer ↔ corpus_eq) -----------------------
    print("\nComputing Faithfulness + Answer Relevancy (sentence-transformers) …")
    from sentence_transformers import SentenceTransformer
    import numpy as np
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    # embed everything
    all_texts_flat = sys_texts + raw_texts + b70_texts + questions + corpus_eqs
    embs = st_model.encode(all_texts_flat, batch_size=32,
                           show_progress_bar=False, normalize_embeddings=True)
    emb_sys = embs[:n]
    emb_raw = embs[n:2*n]
    emb_b70 = embs[2*n:3*n]
    emb_q   = embs[3*n:4*n]
    emb_ceq = embs[4*n:5*n]

    def cos(a, b):
        return float(np.dot(a, b))   # already normalized

    # faithfulness = sim(answer, corpus_eq) — only where corpus_eq non-empty
    faith_sys, faith_b70 = [], []
    faith_mask = []
    for i, ceq in enumerate(corpus_eqs):
        if ceq.strip():
            faith_sys.append(cos(emb_sys[i], emb_ceq[i]))
            faith_b70.append(cos(emb_b70[i], emb_ceq[i]))
            faith_mask.append(i)

    # answer relevancy = sim(answer, question)
    rel_sys = [cos(emb_sys[i], emb_q[i]) for i in range(n)]
    rel_raw = [cos(emb_raw[i], emb_q[i]) for i in range(n)]
    rel_b70 = [cos(emb_b70[i], emb_q[i]) for i in range(n)]

    print(f"  Faithfulness (n={len(faith_mask)}) SYS={_avg(faith_sys):.4f} 70B={_avg(faith_b70):.4f}")
    print(f"  Ans.Relevancy SYS={_avg(rel_sys):.4f} RAW={_avg(rel_raw):.4f} 70B={_avg(rel_b70):.4f}")

    # -- Per-question records ------------------------------------------------
    per_q = []
    for i, r in enumerate(records):
        pq = {
            "id": r["id"], "difficulty": r.get("difficulty","easy"),
            "question": r["question"],
            "bert_f1_sys": round(F_sys[i], 4),
            "bert_f1_raw": round(F_raw[i], 4),
            "rougeL_sys":  round(rl_sys[i], 4),
            "rougeL_raw":  round(rl_raw[i], 4),
            "bleu4_sys":   round(bleu_sys[i], 4),
            "bleu4_raw":   round(bleu_raw[i], 4),
            "ans_rel_sys": round(rel_sys[i], 4),
            "ans_rel_raw": round(rel_raw[i], 4),
            "ans_rel_b70": round(rel_b70[i], 4),
        }
        if i in faith_mask:
            fi = faith_mask.index(i)
            pq["faith_sys"] = round(faith_sys[fi], 4)
            pq["faith_b70"] = round(faith_b70[fi], 4)
        per_q.append(pq)

    # -- Difficulty breakdown ------------------------------------------------
    def _by_diff(key, diff):
        vals = [pq[key] for pq in per_q if pq["difficulty"] == diff and key in pq]
        return round(_avg(vals), 4) if vals else None

    by_diff = {}
    for d in ["easy", "medium", "hard"]:
        sub = [pq for pq in per_q if pq["difficulty"] == d]
        if not sub:
            continue
        by_diff[d] = {k: round(_avg([pq[k] for pq in sub if k in pq]), 4)
                      for k in ["bert_f1_sys","bert_f1_raw","rougeL_sys","rougeL_raw",
                                 "bleu4_sys","bleu4_raw","ans_rel_sys","ans_rel_b70"]}

    # -- Summary -------------------------------------------------------------
    summary = {
        "n": n,
        "reference": "70B (Llama-3.1-70B) — bias noted: 70B is both baseline and reference",
        "overall": {
            "BERTScore_F1": {
                "SYS":  round(_avg(F_sys), 4),
                "RAW":  round(_avg(F_raw), 4),
                "70B_vs_self": 1.0,
            },
            "ROUGE_L": {
                "SYS":  round(_avg(rl_sys), 4),
                "RAW":  round(_avg(rl_raw), 4),
            },
            "BLEU_4": {
                "SYS":  round(_avg(bleu_sys), 4),
                "RAW":  round(_avg(bleu_raw), 4),
            },
            "Faithfulness_vs_corpus": {
                "SYS":  round(_avg(faith_sys), 4),
                "70B":  round(_avg(faith_b70), 4),
                "n_with_corpus_eq": len(faith_mask),
            },
            "Answer_Relevancy": {
                "SYS":  round(_avg(rel_sys), 4),
                "RAW":  round(_avg(rel_raw), 4),
                "70B":  round(_avg(rel_b70), 4),
            },
        },
        "by_difficulty": by_diff,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"summary": summary, "per_question": per_q},
                                   indent=2, ensure_ascii=False), encoding="utf-8")

    # -- Print ---------------------------------------------------------------
    SEP = "-" * 68
    print(f"\n{SEP}")
    print(f"  STAGE 2 — GENERATION QUALITY  (n={n})")
    print(f"  Reference = 70B (bias acknowledged — 70B is reference AND baseline)")
    print(SEP)
    print(f"  {'Metric':<28}{'SYS':>10}{'RAW':>10}{'70B':>10}")
    o = summary["overall"]
    print(f"  {'BERTScore F1':<28}{o['BERTScore_F1']['SYS']:>10.4f}{o['BERTScore_F1']['RAW']:>10.4f}{'(ref)':>10}")
    print(f"  {'ROUGE-L':<28}{o['ROUGE_L']['SYS']:>10.4f}{o['ROUGE_L']['RAW']:>10.4f}{'(ref)':>10}")
    print(f"  {'BLEU-4':<28}{o['BLEU_4']['SYS']:>10.4f}{o['BLEU_4']['RAW']:>10.4f}{'(ref)':>10}")
    print(f"  {'Faithfulness (↑=grounded)':<28}{o['Faithfulness_vs_corpus']['SYS']:>10.4f}{'n/a':>10}{o['Faithfulness_vs_corpus']['70B']:>10.4f}")
    print(f"  {'Answer Relevancy':<28}{o['Answer_Relevancy']['SYS']:>10.4f}{o['Answer_Relevancy']['RAW']:>10.4f}{o['Answer_Relevancy']['70B']:>10.4f}")
    print(SEP)
    for d in ["easy", "medium", "hard"]:
        if d in by_diff:
            bd = by_diff[d]
            print(f"  {d:<8}  BERTScore SYS={bd['bert_f1_sys']:.3f} RAW={bd['bert_f1_raw']:.3f}"
                  f"  |  ROUGE-L SYS={bd['rougeL_sys']:.3f} RAW={bd['rougeL_raw']:.3f}")
    print(SEP)
    print(f"  saved -> {OUT_JSON}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
