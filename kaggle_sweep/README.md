# Kaggle Retrieval Sweep — Stage 3 SOTA hunt

Find the chunking + embedder config that maximizes **Hit@1 / Hit@3 / MRR** for the
physics RAG retriever. Runs in minutes on a P100. No SLM, no API, no LLM.

## Why
Current production retriever (MiniLM-L6, 512-word chunks) gives Hit@3 ≈ 0.47.
This sweep tests smaller chunks + stronger embedders (BGE / E5 / GTE) to push it up.

## Kaggle setup
1. New Notebook → **Settings → Accelerator = GPU P100**, **Internet = ON**.
2. Upload `kaggle_sweep.zip` as a dataset (or via "Add Data" → upload).
3. In a cell:
   ```python
   !cp -r /kaggle/input/<your-dataset-name>/* /kaggle/working/
   %cd /kaggle/working
   !python req.py
   ```
4. Run the sweep:
   ```python
   !python sweep_retrieval_gpu.py --full --rerank
   ```
   Faster subset (4 strong embedders, no large):
   ```python
   !python sweep_retrieval_gpu.py
   ```

## What it does
- Re-chunks `data/corpus/*.txt` at 7 granularities (96 → 512 words).
- Ground truth = `data/evaluation/gold_anchors.json` (each question's true source
  paragraph; a chunk is "gold" if it covers ≥50% of the anchor's content words).
  Chunking-independent and embedder-independent → fair comparison.
- Embeds with each model, retrieves top-3, computes Hit@1/Hit@3/MRR.
- `--rerank` adds a cross-encoder on the winning dense config.

## Send back
`results/sweep_results.json` (or paste the printed LEADERBOARD). That tells me the
winning chunk size + embedder; I rebuild the production index with it.

## Rebuild the production index with the winner (bge-large @ 384/64)
After the sweep confirms the winner, build the real index your teammates' GPU box
(or any GPU) will deploy:
```python
!python rebuild_index_gpu.py
```
This writes `rebuilt_index/`:
```
docs.json  dense.index  bm25_docs.json  registry.json  build_meta.json
```
Download `rebuilt_index/` and drop ALL of it into `backend_new/data/embeddings/`
(replacing the old MiniLM files). The backend config is already pointed at
bge-large (1024-dim, cosine) — see build_meta.json + the CONFIG PATCH the script
prints. Until you swap these files in, the backend will mismatch (1024 vs 384), so
rebuild + swap before running the backend.

## Confirm the live number (hybrid + rerank)
After rebuild, get the deployed-pipeline Hit@k (dense + BM25 + RRF + cross-encoder):
```python
!python confirm_live_gpu.py
```
Prints Hit@1 / Hit@3 / MRR overall + by difficulty. This is the strongest Stage-3
number for the paper (full pipeline, not dense-only). Send me the numbers.

## Files
```
req.py                    # run first: install + preflight
sweep_retrieval_gpu.py    # the sweep (find winner)
rebuild_index_gpu.py      # build production index with winner (bge-large @ 384/64)
confirm_live_gpu.py       # confirm Hit@k on rebuilt index, full hybrid+rerank
requirements.txt
data/corpus/*.txt         # physics corpus
data/evaluation/gold_anchors.json   # ground truth (100 anchors)
```
