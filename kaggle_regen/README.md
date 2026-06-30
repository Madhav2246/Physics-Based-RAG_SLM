# Kaggle Regeneration Package — new bge-large index

The retrieval index changed (MiniLM/512 → **bge-large/384**, Stage-3 Hit@3 0.27 → **0.94**),
so every SYSTEM-generated answer and every downstream physics score is stale.
This package regenerates the whole evaluation chain on the **new index**.

The questions stay fixed (`nvidia_golden_qa.jsonl`); only the SYSTEM's answers are
regenerated. This bundles the new bge-large index + the fine-tuned SLM adapter +
all stage scripts.

## Run (Kaggle GPU P100, Internet ON)
```python
# unzip / add as dataset, then:
%cd /kaggle/working/kaggle_regen/backend_new
!pip install -q -r requirements.txt
%cd /kaggle/working/kaggle_regen
!python run_all.py --smoke      # 5 Q sanity first (fast)
!python run_all.py              # full chain (100 Q)
```
Single step / resume:
```python
!python run_all.py --only 1            # regen answers_dump only
!python run_all.py --from_step 2       # skip stage 1, continue
```

## Chain (order matters)
1. `stage1_physics_new.py` — **GPU**, regenerates `answers_dump.jsonl` + Stage 1 scores (keystone)
2. `stage2_generation.py` — CPU, re-scores dump
3. `stage4_ablation.py` — CPU, re-scores dump
4. `stage4b_validator_test_tempsweep.py` — **GPU**, validator sweep, HARD only,
   **n = 1 3 5 7 9 11 13 15 17** (temp 0.9). Produces the validator-gap-vs-n curve.
   OOM-safe: generation is `num_return_sequences=1` looped n times, so **n=17 uses
   the same GPU memory as n=1** — only runtime scales. Checkpoint/resume per n.
5. `stage6_significance.py` — CPU, significance tests

Optional — Stage-4b TEMPERATURE sweep at fixed n=7 (validator-vs-diversity):
```python
%cd /kaggle/working/kaggle_regen/backend_new
!python scripts/stage4b_validator_test_tempsweep.py --samples 7 --temperature 0.3
!python scripts/stage4b_validator_test_tempsweep.py --samples 7 --temperature 0.9
```
(These overwrite the same per-n files — move outputs between runs to keep both.)

## Send back
Download `backend_new/data/evaluation/` — that holds the new answers_dump + all
stage JSONs. I compare old vs new to quantify "better retrieval → better answers".

## Config baked in
`utils/config.py` already points at bge-large (1024-dim, cosine, query prefix);
`data/embeddings/` is the rebuilt 1539-chunk bge-large index. No edits needed.
```
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"  | EMBED_DIM = 1024
CHUNK 384/64 | IndexFlatIP (cosine) | QUERY_PREFIX set
```
