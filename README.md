# Physics-Based RAG SLM

A domain-specific **Retrieval-Augmented Generation (RAG)** system for semiconductor device physics. The core idea: combine a small language model (SLM) with three deterministic physics validators — symbolic equation parsing, dimensional analysis, and numerical sanity checking — to produce scientifically grounded answers and detect hallucinations automatically.

---

## Architecture

```
Query → Hybrid Retrieval (FAISS + BM25 + CrossEncoder)
      → Prompt Building
      → Multi-Sample SLM Generation (Qwen 2.5-0.5B)
      → Symbolic Equation Parsing (SymPy)
      → Dimensional Analysis
      → Numerical Sanity Check
      → Semantic Similarity (response ↔ evidence)
      → Confidence Scoring
      → Uncertainty Estimation
      → Structured Output + JSONL Logging
```

### Key Technologies

| Layer | Technology |
|---|---|
| Language Model | Qwen 2.5-0.5B-Instruct + LoRA (PEFT) |
| Dense Retrieval | FAISS + `all-MiniLM-L6-v2` |
| Sparse Retrieval | BM25 (rank-bm25) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Symbolic Math | SymPy |
| PDF Extraction | PyMuPDF |
| Training | TRL SFTTrainer + LoRA (r=16, α=32) |

---

## Setup

### 1. Create virtual environment

```bash
python -m venv env_slm
# Windows:
.\env_slm\Scripts\activate
# Linux/Mac:
source env_slm/bin/activate
```

### 2. Install dependencies

```bash
pip install torch transformers accelerate sentence-transformers faiss-cpu \
            rank-bm25 sympy numpy scipy tqdm pymupdf peft trl datasets
```

### 3. Run the demo

```bash
python main.py
```

### 4. Run the 5-query test suite

```bash
python run_tests.py
```

---

## Directory Structure

```
├── main.py                    # Demo entry point
├── run_tests.py               # 5-query test suite
├── requirements.txt
├── pipeline/
│   └── rag_pipeline.py        # 10-step orchestrator
├── reasoning/
│   ├── slm_model.py           # Qwen 2.5 + LoRA inference
│   └── prompt_builder.py      # Evidence → prompt string
├── retrieval/
│   ├── hybrid_retriever.py    # Dense + sparse + rerank
│   ├── dense_retriever.py     # FAISS semantic search
│   ├── sparse_retriever.py    # BM25 keyword search
│   └── reranker.py            # CrossEncoder reranking
├── physics/
│   ├── equation_validator.py  # SymPy equation parsing
│   ├── dimension_checker.py   # Dimensional analysis
│   └── numerical_validator.py # Numerical sanity check
├── utils/
│   ├── config.py              # All configuration constants
│   ├── confidence_engine.py   # Weighted confidence scorer
│   ├── uncertainty_engine.py  # Response stability metric
│   └── logger.py              # JSONL audit logger
├── scripts/
│   ├── extract_pdfs.py        # PDF → corpus text files
│   ├── prepare_data.py        # Corpus → training JSONL
│   └── train_slm.py           # LoRA fine-tuning
├── data/
│   ├── raw_pdfs/              # Source PDFs
│   ├── corpus/                # Extracted text (64 files)
│   ├── processed/             # Training JSONL
│   └── embeddings/            # FAISS index + docs.json
├── finetuned_slm/             # LoRA adapter weights
└── logs/
    └── rag_logs.jsonl         # Query/response audit log
```

---

## Data Pipeline (for re-training)

```bash
# 1. Extract text from PDFs in data/raw_pdfs/
python scripts/extract_pdfs.py

# 2. Chunk corpus into Q&A training format
python scripts/prepare_data.py

# 3. Fine-tune Qwen 2.5 with LoRA (requires GPU)
python scripts/train_slm.py
```

---

## Configuration

All tunable constants are in [`utils/config.py`](utils/config.py):

```python
MODEL_NAME     = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH   = "finetuned_slm"
N_SAMPLES      = 3       # samples for uncertainty estimation
MAX_NEW_TOKENS = 256
TEMPERATURE    = 0.7
TOP_K          = 3       # retrieved documents per query
```

---

## Output Format

Every call to `pipeline.answer(query)` returns:

```python
{
    "response":             str,    # model's generated answer (prompt-stripped)
    "all_responses":        list,   # all N_SAMPLES responses
    "evidence":             list,   # retrieved document chunks
    "symbolic_validation":  str,    # SymPy parse result
    "dimension_validation": str,    # dimensional consistency
    "numerical_validation": str,    # numerical sanity check
    "semantic_similarity":  float,  # cosine sim (answer ↔ evidence)
    "confidence_score":     float,  # 0.0–1.0 weighted score
    "confidence_label":     str,    # HIGH / MODERATE / LOW CONFIDENCE
    "uncertainty_score":    float,  # model stability across samples
    "stability_label":      str,    # HIGH / MODERATE / LOW STABILITY
}
```

---

## Corpus

Training data sourced from:
- MIT OCW 6.012 Semiconductor Device Physics lectures
- S.M. Sze "Physics of Semiconductor Devices" (excerpts)
- BSIM MOSFET model manuals
- IRDS semiconductor roadmaps

64 extracted text files, ~4.6 MB total.
