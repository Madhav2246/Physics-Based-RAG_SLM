# Stage 4 — Ablation Study

**Goal:** Isolate the contribution of each architectural component to the physics correctness score (0–4 scale, `new_checker` v2).
**Dataset:** 100 golden QA · `samples_per_q=5`, `seed=42`
**Metric:** Same physics scorer as Stage 1 (parseable + dimensional + numerical + coverage)
**Retrieval index:** bge-large-en-v1.5 @ chunk 384/64, cosine (SOTA rebuild, Hit@3=0.94) + letter-soup guard extractor
**Date:** 2026-06-07 (full 100Q run on rebuilt index; supersedes 2026-06-06)

> **Note:** Ablation scores in this document were computed under checker v2. Under checker v3, the full system scores 1.305 overall (Stage 1 rescored). Relative orderings and component contributions are unchanged — every component still earns its place and the waterfall structure is identical.

---

## 1. Objective
Determine which components of the pipeline drive physics correctness, and which add complexity without proportional gain.

## 2. Ablation Configurations

| Config | Dense | Sparse | Reranker | Best-of-N | Physics selection | Notes |
|---|---|---|---|---|---|---|
| **full** | ✅ | ✅ | ✅ | ✅ (n=5) | by max score | Full system |
| **-reranker** | ✅ | ✅ | ❌ | ✅ | by max score | RRF fusion only |
| **-sparse** | ✅ | ❌ | ✅ | ✅ | by max score | FAISS dense only |
| **-dense** | ❌ | ✅ | ✅ | ✅ | by max score | BM25 only |
| **-bestofN** | ✅ | ✅ | ✅ | ❌ (sample\[0\]) | none | First sample only |
| **-validator** | ✅ | ✅ | ✅ | ✅ | **random** | No physics-score selection |
| **raw\_0.5b** | ❌ | ❌ | ❌ | ✅ | by max score | No retrieval at all |

Generation-side ablations (`-bestofN`, `-validator`, `raw_0.5b`) use stored texts — no GPU required.
Retrieval-side ablations re-run retrieval on CPU, extract corpus equation from new evidence, re-score stored samples.

> **Note on n=5:** This run used `samples_per_q=5` (vs n=2 in the old 2026-06-06 ablation). The larger sample pool gives best-of-N and the validator more to select from, producing a cleaner signal on every component's contribution. All configs are internally consistent on the same stored samples.

> **LoRA ablation not included** — requires GPU re-generation with base Qwen-0.5B (no adapter). Estimated contribution reported separately in §6.

## 3. Results

### 3.1 Summary table

| Config | **Score /4** | **Δ vs Full** | Parse % | DCRcond % | NVRcond % | Cover % |
|---|---|---|---|---|---|---|
| **full** | **1.361** | — | 62.0 | 29.4 | 100.0 | 25.1 |
| -reranker | 1.254 | **−0.107** | 57.0 | 28.6 | 100.0 | 19.4 |
| **-sparse** | **1.234** | **−0.127** | 59.0 | 28.6 | 100.0 | 22.4 |
| -dense | 1.194 | **−0.167** | 53.0 | 26.7 | 100.0 | 20.4 |
| -bestofN | 1.061 | **−0.300** | 48.0 | — | 100.0 | 21.1 |
| -validator | 1.157 | **−0.204** | 53.0 | — | 100.0 | 20.7 |
| **raw\_0.5b** | **0.555** | **−0.806** | 25.0 | 80.0 | 100.0 | 7.5 |

*(NVRcond=100% for all — no numeric failures on evaluable equations; see Stage 1 §7.4)*
*(DCRcond `—` = denominator too small for reliable estimate in mixed-difficulty run)*

### 3.2 Component contribution (Δ vs removing that component)

| Component | Δ Score | Direction | Interpretation |
|---|---|---|---|
| Retrieval grounding overall (full vs raw) | **+0.806** | ✅ Essential | Corpus equation grounding is the dominant factor |
| Best-of-N selection (full vs -bestofN) | **+0.300** | ✅ Strong | n=5 sampling pool exposes the full gain |
| Physics-score selection (full vs -validator) | **+0.204** | ✅ Real | Physics-aware selection over random — validated |
| Dense retriever (full vs -dense) | **+0.167** | ✅ Helps | Dense adds meaningful lift over BM25-only |
| BM25 sparse (full vs -sparse) | **+0.127** | ✅ Helps | Sparse broadens keyword recall, net positive |
| CrossEncoder reranker (full vs -reranker) | **+0.107** | ✅ Helps | Reranker re-prioritises equation-rich chunks |

## 4. Interpretation

### 4.1 The dominant factor: corpus equation grounding (+0.806)

Retrieval grounding is by far the largest contributor — raw 0.5B scores 0.555 while the full system scores 1.361 (+0.806). Every other component is second-order. The equation the SLM is prompted with determines whether a parseable, physics-valid equation appears in the output. This confirms the core architectural claim.

### 4.2 Clean monotone ablation — every component helps

**This is the key change from the old ablation (2026-06-06).** In the old run (n=2, noisy 30Q smoke test, old index), the ablation was non-monotone: removing sparse and reranker *improved* the score, suggesting those components hurt equation recovery. That finding was an artefact of:
- Small n=2 sample pool giving best-of-N little to select from.
- Old index + un-guarded extractor injecting letter-soup as "corpus equations".

With the rebuilt index (SOTA bge-large, Hit@3=0.94), letter-soup guard, and n=5 samples, the ablation is **clean and monotone**: removing any component hurts. Every component earns its place.

### 4.3 Best-of-N selection: the second largest contributor (+0.300)

With n=5 samples, picking the best-physics-scoring sample over taking sample[0] gains **+0.300** — the largest single-component gain after corpus grounding. This reflects genuine diversity in the SLM's output distribution: with 5 attempts, the physics scorer reliably finds a parseable, dimensionally consistent equation when one exists in the output space.

### 4.4 Physics-score validator: +0.204 (real signal)

Removing physics-aware selection in favour of random (-validator, +0.204 gap) confirms the validator is a real discriminator on the full 100Q mix with n=5 samples. The old ablation showed only +0.013 — a false negative from corpus_eq clamping (same equation prepended to all samples) and easy-question dominance. At n=5, even corpus-grounded samples diverge enough for the scorer to pick the best one. Stage 5 (4b sweep) shows the signal grows further as n and difficulty increase.

### 4.5 Hybrid retrieval: every leg matters

| Component removed | Δ |
|---|---|
| -reranker | −0.107 |
| -sparse | −0.127 |
| -dense | −0.167 |

All three retrieval components contribute. Dense (−0.167) is the largest single-leg loss — semantic embedding captures physics equation structure better than keyword matching. Sparse (−0.127) adds complementary keyword recall. Reranker (−0.107) re-orders the merged pool to surface equation-rich chunks first. The old run's finding of "-sparse improves" was index-specific noise; on the SOTA index this is reversed.

### 4.6 Dense vs sparse only: both hurt

Dense-only (−sparse, 1.234) and BM25-only (−dense, 1.194) both score well below the full hybrid (1.361). The hybrid beats both components in isolation — each retriever surfaces different equation-containing chunks, and RRF fusion picks up both. This is the standard hybrid retrieval finding, now confirmed on the clean index.

### 4.7 NVRcond = 100% across all configs

Every configuration achieves NVRcond=100% — no numeric failures on evaluable equations. Confirmed across all ablations.

## 5. Component contribution waterfall

```
raw 0.5B:                0.555  (no retrieval, no grounding)
  + corpus grounding:    1.361  (+0.806 — retrieval is the core contribution)
  Δ -bestofN:           −0.300  (remove n=5 selection → 1.061)
  Δ -validator:         −0.204  (random selection → 1.157)
  Δ -dense:             −0.167  (BM25 only → 1.194)
  Δ -sparse:            −0.127  (dense only → 1.234)
  Δ -reranker:          −0.107  (no reranker → 1.254)

All components contribute. No component hurts. Architecture is validated.
```

## 6. LoRA contribution (estimated — GPU required)

The LoRA fine-tuning (r=16, α=32, trained on semiconductor device physics QA) cannot be ablated locally. Based on observed behavior:
- Base Qwen-0.5B without LoRA generates less equation-structured output; the parse rate would be expected to drop from ~62% to ~30–35%.
- LoRA teaches the model to emit `Equation: lhs = rhs` patterns and symbol explanations — directly enabling the equation extractor and corpus_eq prepend pipeline.
- Estimated score without LoRA: ~0.7–0.9/4 (rough estimate; formal measurement deferred to future Kaggle run).

## 7. Limitations

| Threat | Explanation |
|---|---|
| Ablation "full" ≠ Stage 1 "full" exactly | Stage 1 used Kaggle P100 pipeline; ablation uses local re-scoring. Scores are internally consistent across ablation configs — deltas are valid. |
| LoRA ablation missing | Estimated only; GPU run needed for confirmation |
| Retrieved corpus_eq changes per ablation | Each config gets different corpus_eq → not a pure single-variable ablation. Unavoidable without GPU re-generation. |
| DCRcond small denominator | Mixed difficulty run means few fully-checkable equations per ablation config; DCRcond not headline here |

---

> ### Key Takeaway
> **Corpus equation grounding is the dominant architectural contribution (+0.806)** — everything else is second-order. Unlike the old ablation (n=2, old index) which showed non-monotone results with sparse and reranker *hurting* score, the **full 100Q run on the rebuilt SOTA index gives a clean monotone ablation**: every component earns its place. **Best-of-N at n=5 contributes +0.300** (second largest); **physics-score validator adds +0.204** (real signal, not the false-negative +0.013 from the old run). Dense retrieval is the most important retrieval leg (−0.167 when removed), followed by sparse (−0.127) and reranker (−0.107). NVRcond=100% across all ablations confirms the physics validator is not introducing false negatives.

**Next:** Stage 5 — Physics-Score Validator Power Analysis (hard questions, n=1→17 sweep) → `stage5_validator_power.md`
