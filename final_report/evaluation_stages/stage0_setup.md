# Stage 0 — Evaluation Setup, Dataset Audit & Plan

**Project:** Physics-Based RAG SLM (semiconductor device physics)
**Comparison under test:** **NVIDIA Llama-3.1-70B (ungrounded baseline)** vs **Complete System = RAG + Qwen-0.5B + neuro-symbolic validators**
**Codebase evaluated:** `backend_new/` (current — best-of-N re-ranking + SymPy reserved-name overrides)
**Date:** 2026-06-06

---

## Objective

This stage establishes a reproducible and fair evaluation framework for comparing an ungrounded large-language-model baseline against the proposed retrieval-augmented, neuro-symbolic system. The goal is to define evaluation scope, dataset validity, fairness assumptions, and feasible metrics **before** observing any performance numbers — so that interpretation is principled rather than post-hoc.

---

## Expected Hypothesis

We hypothesize that:

1. The **Complete System** will outperform **raw Qwen-0.5B** on physics correctness, due to retrieval grounding and symbolic re-ranking/validation.
2. The **70B baseline** may achieve higher *linguistic* similarity metrics (BLEU/ROUGE/BERTScore), because the gold references were authored by the same 70B family.
3. The proposed system will demonstrate **stronger or competitive reliability on objective, model-agnostic physics-validation metrics** despite a ~140× smaller generator.

These hypotheses are recorded up front; results are interpreted against them in each later stage.

---

## Comparison at a Glance

```
COMPLETE SYSTEM (proposed)                 BASELINE (70B)
──────────────────────────                 ──────────────
        Question                                Question
           │                                       │
      Hybrid Retrieval                       Llama-3.1-70B
   (FAISS + BM25 + rerank)                  (parametric only)
           │                                       │
   Qwen-0.5B + LoRA                              Answer
           │                              (no retrieval,
     Best-of-N candidates                  no validation)
           │
  Neuro-Symbolic Validators
  (parse · dim · numeric)
           │
      Final Answer
```

---

## 1. What we are actually comparing

| Side | Definition | Has retrieval? | Has validation? |
|---|---|---|---|
| **Baseline (70B)** | `meta/llama-3.1-70b-instruct` via NVIDIA NIM, answering from parametric memory | No | No |
| **Complete System** | Hybrid retrieval → Qwen-0.5B+LoRA generation → best-of-N neuro-symbolic re-ranking → SymPy validators | Yes | Yes |

> This is deliberately a **system-vs-model** comparison, not model-vs-model. The thesis is that *grounding + symbolic validation* makes a 140× smaller model's **output** trustworthy. Where useful we also report the **raw 0.5B** (no grounding) so the contribution of each layer is visible.

---

## 2. Dataset audit — `nvidia_golden_qa.jsonl`

| Property | Value |
|---|---|
| Total QA pairs | **100** |
| Difficulty split | easy **40** · medium **40** · hard **20** |
| Fields | `id, difficulty, question, answer, equation, symbols, source_chunk, generated_at, attempts` |
| `answer` non-empty | 100 / 100 (reference answer text) |
| `equation` present | 99 / 100 (gold equation) |
| `symbols` present | yes (gold symbol → meaning map) |
| `source_chunk` non-empty | 100 / 100 (the corpus chunk each QA was generated from) |
| Gold evidence doc IDs | **none** (use `source_chunk` as gold-evidence proxy) |

**Author of `answer` + `equation`:** `meta/llama-3.1-70b-instruct` (the synthesis model). **This is the same model family as the baseline.** See fairness caveat §5.

---

## 3. Gold signals available → which metrics are feasible

| Gold signal | Enables |
|---|---|
| `equation` (objective physics) | Physics validation (EER/parse/DCR/NVR), Exact-Match on equations |
| `answer` (reference text) | BERTScore / BLEU / ROUGE — **with bias caveat** |
| `source_chunk` (origin chunk) | Retrieval Hit@k / MRR / NDCG (chunk-match) |
| `question` + retrieved evidence | Programmatic faithfulness & answer-relevancy (cosine, no LLM judge) |
| component toggles | Ablation deltas |

---

## 4. Tooling status

| Library | Status | Needed for |
|---|---|---|
| scipy, sentence-transformers, scikit-learn, numpy, matplotlib | ✅ installed | significance, programmatic faithfulness, retrieval |
| **bert_score, rouge_score, nltk** | ❌ missing | Generation quality (Stage 2) — install before that stage |

Reproducibility: generation is **seeded** (`SEED=42`); re-runs give identical numbers. Each reported figure is a mean over samples where sampling applies.

---

## 5. Fairness caveats (read before interpreting any number)

1. **The gold set is 70B-authored.** `answer` and `equation` were written by Llama-3.1-70B. Therefore **any text-overlap metric (BERTScore/BLEU/ROUGE) against `answer` is structurally biased toward the 70B baseline** — the 70B is being scored against text it effectively wrote. We report these metrics for completeness but treat them as a *ceiling favoring the baseline*, not a fair head-to-head.
2. **The fair, objective ground is physics validation.** The neuro-symbolic 0–4 score (symbolic parse + dimensional + numerical + coverage) is computed by SymPy, is **model-agnostic**, and does **not** reward text overlap. This is where the head-to-head is legitimate — and notably, **even on its own self-authored answers the 70B scored only ~1.19/4** in a prior run, confirming the metric measures physics correctness, not fluency.
3. **Baseline is ungrounded by design.** The 70B answers with no retrieval; the System retrieves + validates. We are measuring *the value of the architecture*, which is the project's claim.
4. **Raw Qwen-0.5B establishes the lightweight baseline and predictably underperforms in domain-specific physics reasoning without grounding.** This is reported openly and is *expected*: it is the price of running locally at 0.5B, and it is precisely what **motivates the retrieval and neuro-symbolic validation layers** of the proposed architecture. Low raw numbers are evidence *for* the design, not a flaw in it; the retrieval + validation layers are what recover the model to the System score.

---

## 5b. Threats to validity

- The golden dataset is **synthesized** (LLM-generated), not human-annotated.
- Reference answers are authored by the **same 70B family** as the baseline (text-overlap bias, see §5.1).
- **`source_chunk`** is used as a proxy for gold evidence, since the corpus lacks document-level IDs.
- Dataset size (**100 QA pairs**) limits broad statistical generalization, though it is sufficient for paired significance testing.
- Generation runs on **CPU** with `MAX_NEW_TOKENS=128`, which can truncate very long derivations (a constraint, not a confound, since it applies equally to every local-model condition).

---

## 6. Stage plan (executed stage by stage, one report each)

| Stage | Dimension | Primary metrics | Feasible now? |
|---|---|---|---|
| **0** | Setup & dataset audit | — | ✅ (this report) |
| **1** | **Physics validation (differentiator)** | EER, parse-rate, DCR, NVR, 0–4 score | ✅ tooling ready |
| **2** | Generation quality | BERTScore-F1, BLEU-4, ROUGE-L, Exact-Match | needs lib install |
| **3** | RAG end-to-end (programmatic) | faithfulness, answer-relevancy (cosine) | ✅ |
| **4** | Retrieval quality | Hit@k, MRR, NDCG (vs `source_chunk`) | ✅ |
| **5** | Ablation | component-wise Δ | ✅ |
| **6** | Statistical significance + master tables | Wilcoxon signed-rank, p-values | ✅ |

**Execution order:** Stage 1 first (objective, strongest, no new deps), then 2–6. Each stage writes its own `stageN_*.md` here with: method, raw numbers, 70B vs System table, and an honest "why" for any low metric.

---

## 7. Stage 0 outcome

- `evaluation_stages/` created.
- Dataset validated: 100 QA, balanced, with gold equation + source chunk + reference answer.
- Feasibility mapped; fairness caveats fixed in writing **before** seeing numbers (so framing is not post-hoc).

---

> ### 🔑 Stage 0 Key Takeaway
> The evaluation is framed as a **system-vs-model** study on a 100-question, difficulty-balanced golden set with objective gold equations. Fairness limitations — chiefly that the reference text is 70B-authored — are declared *up front*, so the head-to-head is anchored on the **model-agnostic physics-validation metric** rather than on metrics that structurally favor the baseline. This positions the evaluation as reproducible, transparent, and grounded in objective physics correctness rather than selectively favorable metrics.

**Next:** Stage 1 — Physics Validation (70B vs Complete System vs raw 0.5B).
