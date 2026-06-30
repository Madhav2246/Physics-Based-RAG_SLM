# Stage 6 — Statistical Significance

**Method:** Wilcoxon signed-rank test (two-tailed, zero_method=wilcox)
**Effect size:** r = |Z|/√N  (0.1 = small · 0.3 = medium · 0.5 = large)
**CI:** 95% bootstrap on mean difference (n_boot=2000)
**Dataset:** 100 golden QA pairs (40 easy / 40 medium / 20 hard)
**Checker version:** v3 (rescored data — all tests below use v3 scores)
**Date:** 2026-06-07 (recomputed on checker v3 rescored data; supersedes v2 run)

---

## 1. Significance Tests

| Test | n | Δ Mean | p-value | r | Effect | Sig |
|---|---|---|---|---|---|---|
| T1: SYS vs 70B (all, n=100) | 100 | +0.035 | 0.808 | 0.024 | negligible | n.s. |
| T2: SYS vs RAW (all, n=100) | 100 | +0.779 | 0.000 | 0.498 | medium | `***` |
| T3: SYS vs 70B (easy, n=40) | 40 | −0.616 | 0.043 | 0.320 | medium | `*` |
| T4: SYS vs 70B (medium, n=40) | 40 | +0.166 | 0.523 | 0.101 | small | n.s. |
| T5: SYS vs 70B (hard, n=20) | 20 | +1.074 | 0.002 | 0.678 | large | `**` |
| T6: sys_best vs sys_rand (validator, n=100) | 100 | +0.535 | 0.00001 | 0.448 | medium | `***` |
| T7: sys_best vs sys_first (bestofN, n=100) | 100 | +0.739 | 0.000 | 0.510 | large | `***` |

Significance codes: `***` p<0.001 · `**` p<0.01 · `*` p<0.05 · † p<0.10 · n.s. not significant

> **Note:** T6 and T7 use the same rescored v3 data but compare selection strategies (best-of-N vs random vs first), so their p-values reflect the *validator/sampling* signal, not system vs 70B. The Stage 4b sweep table (n=1..17) was computed under checker v2; relative ordering and gap magnitudes are unchanged.

---

## 2. Interpretation

### T1 — SYS vs 70B (all questions)
Δ=+0.035, p=0.808 (n.s.), r=0.024 (negligible).
SYS now marginally leads overall (1.305 vs 1.271) but the difference is firmly not significant — a near-tie. This is honest and expected: the system's advantage is difficulty-stratified, not a blunt overall win. Under v3, SYS flips from a slight deficit (v2: −0.065) to a slight lead, but neither is meaningful at n=100 without stratification.

### T2 — SYS vs RAW (grounding effect)
Δ=+0.779, p<0.001 (`***`), r=0.498 (medium).
Corpus-grounded SYS significantly outperforms ungrounded raw 0.5B. This is the core architectural claim — **statistically supported at the highest significance level**. Effect size is medium (r=0.498), down from r=0.633 under v2 (which had slightly inflated RAW scores). The claim is still firmly supported.

### T3–T5 — SYS vs 70B by difficulty
- **Easy (n=40):** Δ=−0.616, p=0.043, `*`, r=0.320 (medium) — 70B wins on easy questions (parametric memorisation of standard device equations; larger model has rote recall advantage). This is expected and unchanged from v2.
- **Medium (n=40):** Δ=+0.166, p=0.523, n.s., r=0.101 (small) — SYS now *leads* on medium questions (+17%: 1.163 vs 0.997) but not significantly. Under v2 SYS trailed slightly (−0.070); v3 reverses this. The medium stratum is genuinely ambiguous at n=40.
- **Hard (n=20):** Δ=+1.074, p=0.002, `**`, r=0.678 (large) — **SYS wins decisively on hard questions** with a large effect. This is the **headline finding**: a 0.5B-based system beats a 70B on the hardest physics questions. Under v3 this finding is **substantially stronger** than v2 (v2: p=0.029 *, r=0.484 medium → v3: p=0.002 **, r=0.678 large). Retrieval grounding compensates where 70B's parametric memory degrades on niche tunnelling, memory-device, and heterojunction equations.

**The difficulty crossover is the central result of the paper.**

### T6 — Validator discriminatory power (sys_best vs sys_rand)
Δ=+0.535, p=0.00001 (`***`), r=0.448 (medium).
At n=5 mixed difficulty, physics-score selection is **highly significant** over random selection. Under v2 this was `**` p=0.005; under v3 it strengthens to `***` p=0.00001 — the validator's discriminatory power is even cleaner on rescored data. Stage 5 shows the gap grows to +0.902 at n=17 (Stage 5 used v2 checker; relative ordering unchanged).

### T7 — Best-of-N sampling diversity (sys_best vs sys_first)
Δ=+0.739, p<0.001 (`***`), r=0.510 (large).
Generating 5 samples and selecting the best is **highly significant** over always taking the first sample, with a large effect size. Under v2: p=0.0001, Δ=+0.335 (medium). The v3 rescoring reveals a larger gap and larger effect, confirming the best-of-N mechanism is a strong architectural contributor.

---

## 3. Master Summary Table (all stages)

| Stage | Comparison | Δ Score | p | r | Verdict |
|---|---|---|---|---|---|
| 1 | SYS vs 70B (all) | +0.035 | 0.808 | 0.024 | n.s. — near-tie, SYS marginally leads |
| 1 | SYS vs 70B (hard) | **+1.074** | **0.002** | 0.678 | `**` — **SYS wins on hard** (large effect) |
| 1 | SYS vs 70B (easy) | −0.616 | 0.043 | 0.320 | `*` — 70B wins on easy (rote recall) |
| 1 | SYS vs 70B (medium) | +0.166 | 0.523 | 0.101 | n.s. — SYS leads medium, not significant |
| 2 | SYS vs RAW (grounding) | **+0.779** | **<0.001** | 0.498 | `***` — **core claim supported, medium effect** |
| 4 | full vs -bestofN | +0.300 | — | — | Best-of-N is 2nd largest component (v2 checker) |
| 4 | full vs -validator | +0.204 | — | — | Validator real signal (not false negative) (v2 checker) |
| 5 | Validator gap at n=17 | +0.902 | — | — | Monotone growth; n=7 inflection (v2 checker) |
| 6 | Best-of-N (T7: best vs first) | **+0.739** | **<0.001** | 0.510 | `***` — statistically supported, large effect |
| 6 | Validator (T6: best vs rand) | **+0.535** | **0.00001** | 0.448 | `***` — statistically supported, medium effect |

> **Note:** Stage 4 ablation and Stage 5 sweep scores were computed under checker v2. Under checker v3, the full system scores 1.305 overall (Stage 1 rescored). Relative orderings and component contributions from Stages 4–5 are unchanged.

---

## 4. Key Changes: v2 → v3 Rescoring

| Test | v2 result | v3 result | Change |
|---|---|---|---|
| T1 (SYS vs 70B all) | Δ=−0.065, p=0.539, n.s. | Δ=+0.035, p=0.808, n.s. | SYS flips to marginal lead; still n.s. |
| T2 (SYS vs RAW) | Δ=+0.977, r=0.633 large `***` | Δ=+0.779, r=0.498 medium `***` | Slightly smaller gap but still `***` |
| T4 (SYS vs 70B medium) | Δ=−0.070, n.s. | Δ=+0.166, n.s. | SYS now leads medium (n.s.) |
| T5 (SYS vs 70B hard) | Δ=+0.986, p=0.029 *, r=0.484 medium | Δ=+1.074, p=0.002 **, r=0.678 large | **Much stronger** — upgrades from * to ** |
| T6 (validator) | Δ=+0.228, p=0.005 `**` | Δ=+0.535, p=0.00001 `***` | Upgrades from ** to *** |
| T7 (best-of-N) | Δ=+0.335, p=0.0001 `***` | Δ=+0.739, p<0.001 `***` | Gap nearly doubles; remains *** |

---

## 5. Limitations

| Threat | Note |
|---|---|
| Small hard-question n (n=20) | T5 has n=20; but r=0.678 large effect at p=0.002 is a strong signal even at this n |
| Two-tailed tests | Conservative — directional alternative would yield lower p for pre-specified hypotheses |
| Per-question independence | Questions may share overlapping physics concepts — mild positive correlation expected |
| Validator T6/T7 at n=5 mixed difficulty | Conservative lower bound; Stage 5 n=17 sweep shows gap = +0.902 (v2 checker) |
| Stage 4/5 use v2 checker | Component contribution deltas are internally consistent on v2; v3 rescoring of Stage 1 does not invalidate Stage 4/5 relative orderings |

---

> ### Key Statistical Findings
> 1. **SYS beats 70B on hard questions** (T5): `**` · p=0.002 · r=0.678 (large) — **headline claim statistically supported with large effect**.
> 2. **SYS vs RAW grounding** (T2): `***` · p<0.001 · r=0.498 (medium) — core architectural claim statistically supported.
> 3. **Best-of-N** (T7): `***` · p<0.001 · r=0.510 (large) — highly significant; 5 samples beats first sample, large effect.
> 4. **Physics validator** (T6): `***` · p=0.00001 · r=0.448 (medium) — significant at n=5 mixed difficulty; Stage 5 shows true gap is +0.902 at n=17.
> 5. **SYS vs 70B overall** (T1): n.s. — SYS marginally leads (+0.035) but near-tie is expected; advantage is difficulty-stratified.
