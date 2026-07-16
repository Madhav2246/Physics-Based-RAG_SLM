# SLM Physics RAG Engine — Final Evaluation Strategy (v2.0)
## Publication-Ready Evaluation Framework

---

# 1. Evaluation Dimensions Overview

| # | Dimension | Key Metrics | Priority |
|---|-----------|------------|----------|
| 1 | **Retrieval Quality** | Hit Rate@k, MRR, NDCG | 🔴 Critical |
| 2 | **Generation Quality** | BERTScore F1, BLEU-4, ROUGE-L | 🔴 Critical |
| 3 | **Physics Validation** ⭐ | EER, DCR, NVR, Hallucination Detection Rate | 🔴 Your differentiator |
| 4 | **RAG End-to-End** | Faithfulness, Answer Relevancy (programmatic) | 🟡 Important |
| 5 | **Ablation Study** | Component-wise delta measurement | 🟡 Important |
| 6 | **Baselines + Statistical Significance** | p-values via Wilcoxon signed-rank test | 🔴 Critical |

---

# 2. Retrieval Quality Metrics

| Metric | Formula | What It Proves |
|--------|---------|---------------|
| **Hit Rate @k** | `1 if gold_doc ∈ top_k else 0` | Correct doc is found |
| **MRR** | `1 / rank_of_first_relevant` | Correct doc is ranked high |
| **NDCG@k** | Normalized Discounted Cumulative Gain | Overall ranking quality |
| **Context Precision** | Relevant chunks in top positions / total top positions | Retriever precision |
| **Context Recall** | Retrieved relevant / total relevant in corpus | Retriever completeness |

**Requirement**: 50+ queries with manually labeled gold evidence documents.

---

# 3. Generation Quality Metrics

| Metric | Type | Why This One |
|--------|------|-------------|
| **BERTScore F1** | Semantic | **Primary metric** — handles paraphrasing, symbol reordering (`μ·Cox` vs `Cox·μ`) |
| **BLEU-4** | Lexical (n-gram precision) | Standard baseline metric for comparison |
| **ROUGE-L** | Lexical (LCS recall) | Captures longest common subsequence |
| **Exact Match** | Exact | For equation-only extraction tasks |

> [!IMPORTANT]
> **Use standard library implementations, not heuristics.** Reviewers will verify mathematical compliance.

```python
# REQUIRED: Standard implementations for paper
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def compute_bleu4(prediction, reference):
    """Standard BLEU-4 with smoothing (Papineni et al., 2002)"""
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)

def compute_rouge_l(prediction, reference):
    """Standard ROUGE-L (Lin, 2004)"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure

def compute_bertscore(predictions, references):
    """BERTScore F1 (Zhang et al., 2020)"""
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    return F1.tolist()
```

---

# 4. Physics-Specific Validation Metrics (YOUR UNIQUE CONTRIBUTION)

This is what no other RAG paper has. Frame it as a **neuro-symbolic safety layer**.

| Metric | Formula | What It Catches |
|--------|---------|----------------|
| **Equation Extraction Rate (EER)** | `extracted / total_equation_queries` | Can the system find equations in SLM output? |
| **Symbolic Parse Success Rate** | `parsed / extraction_attempts` | Does SymPy successfully parse the equation? |
| **Dimensional Correctness Rate (DCR)** | `dim_correct / total_parsed` | Are LHS/RHS units consistent? |
| **Numerical Validity Rate (NVR)** | `realistic_values / total_evaluated` | Do substituted values fall in physical bounds? |
| **Hallucination Detection Rate (HDR)** | `caught / total_hallucinations` | How many wrong equations did validators flag? |
| **False Positive Rate (FPR)** | `false_flags / total_correct_equations` | How often do validators wrongly reject a good equation? |
| **Composite Physics Score** | `0.25×EER + 0.25×DCR + 0.25×NVR + 0.25×HDR` | Single aggregate physics score |

```python
def physics_metrics(results):
    """Compute all physics-specific metrics"""
    eq_queries = [r for r in results if r["has_equation"]]
    
    # EER
    extracted = sum(1 for r in eq_queries if r.get("equation_found"))
    eer = extracted / max(len(eq_queries), 1)
    
    # Parse Rate
    parsed = [r for r in results if "✔" in r.get("symbolic_validation", "")]
    parse_rate = len(parsed) / max(extracted, 1)
    
    # DCR
    dim_correct = sum(1 for r in parsed if "✔" in r.get("dimension_validation", ""))
    dcr = dim_correct / max(len(parsed), 1)
    
    # NVR
    num_valid = sum(1 for r in parsed if "✔" in r.get("numerical_validation", ""))
    nvr = num_valid / max(len(parsed), 1)
    
    # HDR
    hallucinated = [r for r in results if r.get("is_hallucination")]
    caught = sum(1 for r in hallucinated if r.get("confidence_score", 1.0) < 0.6)
    hdr = caught / max(len(hallucinated), 1)
    
    # FPR
    correct_eqs = [r for r in results if r.get("is_correct_equation")]
    false_flags = sum(1 for r in correct_eqs if r.get("confidence_score", 1.0) < 0.6)
    fpr = false_flags / max(len(correct_eqs), 1)
    
    cps = 0.25 * eer + 0.25 * dcr + 0.25 * nvr + 0.25 * hdr
    
    return {
        "EER": round(eer, 4), "Parse_Rate": round(parse_rate, 4),
        "DCR": round(dcr, 4), "NVR": round(nvr, 4),
        "HDR": round(hdr, 4), "FPR": round(fpr, 4),
        "CPS": round(cps, 4)
    }
```

---

# 5. RAG End-to-End — Solving the LLM-as-a-Judge Bottleneck

> [!WARNING]
> **Problem**: RAGAS uses GPT-4 as judge. Using a massive closed-source model to evaluate your SLM creates a dependency contradiction in a paper about small, local models.

### Solution: Programmatic Faithfulness & Relevancy (No External LLM)

```python
from sentence_transformers import SentenceTransformer, util

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def programmatic_faithfulness(response, evidence_list):
    """Measures if response is grounded in evidence (no LLM judge needed)"""
    sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    evidence_text = " ".join(evidence_list)
    ev_emb = embed_model.encode(evidence_text)
    scores = []
    for sent in sentences:
        sent_emb = embed_model.encode(sent)
        sim = float(util.cos_sim(sent_emb, ev_emb)[0][0])
        scores.append(sim)
    return sum(scores) / len(scores)

def programmatic_answer_relevancy(response, query):
    """Measures if response addresses the query (no LLM judge needed)"""
    q_emb = embed_model.encode(query)
    r_emb = embed_model.encode(response)
    return float(util.cos_sim(q_emb, r_emb)[0][0])
```

**Methodology note for paper**: "We employ programmatic faithfulness and relevancy metrics using cosine similarity of contextual embeddings (all-MiniLM-L6-v2), avoiding dependency on external closed-source LLMs for evaluation. This ensures our evaluation pipeline is fully reproducible and self-contained."

---

# 6. Ablation Study Design

| Experiment | What's Removed | Metrics to Report |
|-----------|---------------|-------------------|
| **Full Pipeline** | Nothing (baseline) | All metrics |
| **– Dense Retriever** | FAISS removed, BM25 only | Hit Rate, MRR, BERTScore |
| **– Sparse Retriever** | BM25 removed, FAISS only | Hit Rate, MRR, BERTScore |
| **– Cross-Encoder Reranker** | Skip reranking | Hit Rate, MRR |
| **– LoRA Fine-tuning** | Base Qwen 0.5B, no adapter | BERTScore, Faithfulness, DCR |
| **– Equation Validator** | Skip SymPy parsing | HDR drops to 0% |
| **– Dimension Checker** | Skip dimensional analysis | DCR drops to N/A |
| **– Numerical Validator** | Skip numerical bounds | NVR drops to N/A |
| **– Multi-Sample (n=1)** | Single response only | Uncertainty unavailable |

---

# 7. Baseline Comparisons

| Baseline | Setup | Purpose |
|----------|-------|---------|
| **Base Qwen 0.5B (no RAG, no LoRA)** | Direct inference | Proves RAG + fine-tuning adds value |
| **Base Qwen 0.5B + RAG (no LoRA)** | Retrieval but no adapter | Proves LoRA helps |
| **Qwen 0.5B + LoRA (no RAG)** | Fine-tuned but no retrieval | Proves retrieval helps |
| **GPT-4o (zero-shot)** | API call, same questions | Shows domain-specific SLM competitiveness |
| **GPT-4o + same RAG context** | API with same evidence | Isolates model quality difference |

---

# 8. Statistical Significance Testing

> [!IMPORTANT]
> Top-tier reviewers (ACL/NeurIPS) require proof that improvements aren't flukes.

```python
from scipy.stats import wilcoxon, ttest_rel

def significance_test(scores_full, scores_ablated, metric_name):
    """Wilcoxon signed-rank test (non-parametric, paired)"""
    stat, p_value = wilcoxon(scores_full, scores_ablated)
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
    return {"metric": metric_name, "p_value": round(p_value, 6), "significance": sig}

# Usage: compare per-query BERTScores between full pipeline and ablated version
# result = significance_test(full_bertscores, no_lora_bertscores, "BERTScore")
```

**In your paper tables**, annotate with `*` (p<0.05), `**` (p<0.01), `***` (p<0.001).

---

# 9. Gold Standard Test Set Structure (50 Queries)

| Category | Count | Examples |
|----------|-------|---------|
| **Equation retrieval** | 15 | Drain current, body effect, subthreshold swing, transconductance |
| **Conceptual** | 15 | Temperature effects, operating regions, channel length modulation |
| **Multi-hop reasoning** | 10 | "If Vgs increases and temperature rises, what happens to Id?" |
| **Out-of-domain** | 5 | "What is the capital of France?" — tests rejection ability |
| **Adversarial** | 5 | Intentionally misleading physics questions — tests robustness |

Each entry requires:
```json
{
  "query": "...",
  "gold_answer": "...",
  "gold_evidence": ["doc1", "doc2"],
  "has_equation": true,
  "gold_equation": "Id = 0.5*mu*Cox*(W/L)*(Vgs-Vth)^2",
  "gold_dimension": {"I": 1},
  "is_hallucination": false,
  "is_correct_equation": true,
  "category": "equation"
}
```

---

# 10. Results Tables for Paper

### Table 1: Retrieval Performance
| Config | Hit Rate@3 | MRR | NDCG@3 |
|--------|-----------|-----|--------|
| BM25 Only | — | — | — |
| FAISS Only | — | — | — |
| Hybrid (no rerank) | — | — | — |
| **Hybrid + Reranker** | — | — | — |

### Table 2: Generation Quality
| Config | BLEU-4 | ROUGE-L | BERTScore F1 | Faithfulness |
|--------|--------|---------|-------------|-------------|
| Base Qwen 0.5B | — | — | — | — |
| + RAG | — | — | — | — |
| + LoRA | — | — | — | — |
| **+ RAG + LoRA** | — | — | — | — |
| GPT-4o (zero-shot) | — | — | — | — |

*Statistically significant improvements marked with \* (p<0.05), \*\* (p<0.01), \*\*\* (p<0.001) via Wilcoxon signed-rank test.*

### Table 3: Physics Validation (Novel Contribution)
| Metric | Without Validators | With Validators | Δ |
|--------|-------------------|----------------|---|
| Hallucination Detection Rate | 0% | —% | +—% |
| Dimensional Correctness Rate | N/A | —% | — |
| Numerical Validity Rate | N/A | —% | — |
| False Positive Rate | N/A | —% | — |
| Composite Physics Score | 0.00 | — | +— |

---

# 11. Papers & Benchmarks to Cite

### RAG Evaluation
| Paper | Venue | Why Cite |
|-------|-------|---------|
| RAGAS (Es et al., 2023) | arXiv:2309.15217 | Standard RAG eval framework |
| RAG Eval Survey (2025) | arXiv:2504.14891 | Comprehensive 2025 survey |
| ARES | ACL 2024 | Automated RAG evaluation |

### Scientific / Physics
| Reference | Venue | Why Cite |
|-----------|-------|---------|
| PHYSICSEVAL (19,609 problems) | ACL 2024 | Large-scale physics QA benchmark |
| LLM-SRBench | NeurIPS 2025 | Scientific equation discovery |
| FEABench | NeurIPS 2025 | Physics/engineering AI benchmark |
| MMLU-Physics subset | Standard | Domain knowledge baseline |

### Neuro-Symbolic & Metrics
| Reference | Venue | Why Cite |
|-----------|-------|---------|
| BERTScore (Zhang et al., 2020) | ICLR 2020 | Semantic similarity metric |
| Neuro-Symbolic ALP for hallucination | MDPI 2025 | Your SymPy validation is this paradigm |
| BLEU (Papineni et al., 2002) | ACL 2002 | Standard generation metric |
| ROUGE (Lin, 2004) | ACL W 2004 | Standard summarization metric |

---

# 12. Action Plan

| Step | Action | Time |
|------|--------|------|
| 1 | Fix bugs from audit (prompt leakage, uncertainty inversion, symbol dict, LoRA path) | 2–3h |
| 2 | Build 50-query gold standard test set from your corpus textbooks | 3–4h |
| 3 | Install eval libs: `pip install nltk rouge_score bert_score scipy` | 5 min |
| 4 | Implement full evaluation script (adapt code above) | 2h |
| 5 | Run retrieval metrics across all 4 retriever configs | 1h |
| 6 | Run generation metrics across all 5 model configs | 2h |
| 7 | Run physics validation metrics | 1h |
| 8 | Run ablation study (9 experiments) | 3–4h |
| 9 | Compute statistical significance for all comparisons | 30 min |
| 10 | Compile Tables 1–3 for paper | 1h |
| | **Total** | **~16–18 hours** |
