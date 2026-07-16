import os

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH  = os.path.join("models", "finetuned_slm")  # where train_slm.py saves adapter

# ── Generation ────────────────────────────────────────────────────────────────
N_SAMPLES      = 3      # 3 samples for robust uncertainty / confidence estimation
MAX_NEW_TOKENS = 128    # capped at 128 to prevent slow CPU runs on repetitive loops
TEMPERATURE    = 0.3    # lower = more coherent for 0.5B; 0.7 causes hallucinations
TOP_P          = 0.9

# ── Reproducibility (Tier 1) ────────────────────────────────────────────────────
# SEED is applied right before every generation. With a fixed int, re-running an
# eval produces the SAME outputs (and therefore the same physics scores), so the
# headline number stops bouncing between runs. Set to None to restore fully
# stochastic generation.
SEED               = 42
# How many samples to draw per question when measuring the score *distribution*
# (mean ± std). A single sample is a coin flip; averaging several is stable.
EVAL_SAMPLES_PER_Q = 5

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K           = 3
EMBEDDINGS_PATH = "data/embeddings"

# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_FILE         = os.path.join("logs", "rag_logs.jsonl")
CORPUS_DIR       = os.path.join("data", "corpus")
TRAIN_DATA_PATH  = os.path.join("data", "processed", "train_dataset.jsonl")
FINETUNE_OUT_DIR = os.path.join("models", "finetuned_slm")      # must match ADAPTER_PATH above