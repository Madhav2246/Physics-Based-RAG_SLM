"""
evaluate_candidate.py
─────────────────────
Evaluation gate for the HITL training pipeline.

This script runs the 10 hand-crafted holdout physics questions against:
 1. The current production adapter (models/finetuned_slm) or base model
 2. The newly trained candidate adapter (models/candidate_slm)

It compares their semantic similarity to the reference answers using the
SentenceTransformer model in our retriever.
If the candidate model achieves a higher or equal average semantic similarity
score than the current production model, it is "promoted" (moved to
models/finetuned_slm). Otherwise, it is discarded.
"""

import json
import shutil
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.rag_pipeline import RAGPipeline

HOLDOUT_SET = PROJECT_ROOT / "data" / "evaluation" / "hitl_eval_holdout.json"
CANDIDATE_DIR = PROJECT_ROOT / "models" / "candidate_slm"
PROD_DIR = PROJECT_ROOT / "models" / "finetuned_slm"
LIVE_FILE = PROJECT_ROOT / "data" / "feedback" / "live_training.json"


def evaluate_adapter(pipeline: RAGPipeline, dataset: list[dict], name: str) -> tuple[float, float]:
    print(f"\n[EVAL] Running 10 holdout questions on {name}...")
    total_sim = 0.0
    total_kw_match = 0.0
    
    for i, item in enumerate(dataset):
        q = item["question"]
        ref = item["reference_answer"]
        keywords = item.get("keywords", [])
        
        # Answer the question
        result = pipeline.answer(q)
        ans = result["response"]
        
        # Calculate semantic similarity to the reference answer
        emb_ref = pipeline.retriever.dense.embed_model.encode(ref, convert_to_tensor=True)
        emb_ans = pipeline.retriever.dense.embed_model.encode(ans, convert_to_tensor=True)
        import torch
        sim = torch.nn.functional.cosine_similarity(emb_ref, emb_ans, dim=0).item()
        
        # Keyword check (factual accuracy proxy)
        ans_lower = ans.lower()
        if keywords:
            matched = sum(1 for kw in keywords if kw.lower() in ans_lower)
            kw_score = matched / len(keywords)
        else:
            kw_score = 1.0  # No keywords means automatic pass for this check
            
        print(f"  [{i+1}/10] Sim: {sim:.3f} | KW: {kw_score:.0%} | {q[:55]}...")
        total_sim += sim
        total_kw_match += kw_score
        
    avg_sim = total_sim / len(dataset)
    avg_kw  = total_kw_match / len(dataset)
    print(f"[{name.upper()} SCORE] Sim: {avg_sim:.3f} | KW Match: {avg_kw:.0%}")
    return avg_sim, avg_kw


def update_live_log(msg: str):
    if LIVE_FILE.exists():
        try:
            data = json.loads(LIVE_FILE.read_text(encoding="utf-8"))
            data.setdefault("log", []).append(msg)
            LIVE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass


def main():
    if not HOLDOUT_SET.exists():
        print(f"[FAIL] Holdout set not found at {HOLDOUT_SET}")
        return
        
    if not CANDIDATE_DIR.exists():
        print(f"[FAIL] Candidate adapter not found at {CANDIDATE_DIR}. Run train_from_feedback.py first.")
        return

    with open(HOLDOUT_SET, "r", encoding="utf-8") as f:
        holdout_data = json.load(f)
        
    print("="*60)
    print("🚀 STARTING EVALUATION GATE")
    print("="*60)
    
    # 1. Evaluate Current Production Model
    print("\nLoading current production pipeline (Base + Production Adapter if exists)...")
    prod_pipeline = RAGPipeline()
    prod_sim, prod_kw = evaluate_adapter(prod_pipeline, holdout_data, "Production Model")
    
    # Free up memory
    del prod_pipeline
    import torch
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    # 2. Evaluate Candidate Model
    print("\nLoading candidate pipeline (Base + Candidate Adapter)...")
    # Temporarily point the pipeline to the candidate directory
    # We do this by instantiating the pipeline and then manually swapping the model
    # Wait, RAGPipeline has hardcoded paths. Let's just swap the directory temporarily.
    
    TEMP_BACKUP_DIR = PROJECT_ROOT / "models" / "finetuned_slm_backup_tmp"
    
    # Move PROD to BACKUP
    if PROD_DIR.exists():
        PROD_DIR.rename(TEMP_BACKUP_DIR)
        
    # Move CANDIDATE to PROD (so RAGPipeline loads it)
    CANDIDATE_DIR.rename(PROD_DIR)
    
    try:
        candidate_pipeline = RAGPipeline()
        cand_sim, cand_kw = evaluate_adapter(candidate_pipeline, holdout_data, "Candidate Model")
        del candidate_pipeline
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        # Revert
        if PROD_DIR.exists():
            PROD_DIR.rename(CANDIDATE_DIR)
        if TEMP_BACKUP_DIR.exists():
            TEMP_BACKUP_DIR.rename(PROD_DIR)
        return

    # 3. Decision Gate
    print("\n" + "="*60)
    print(f"Production Score: Sim {prod_sim:.3f} | KW {prod_kw:.0%}")
    print(f"Candidate Score:  Sim {cand_sim:.3f} | KW {cand_kw:.0%}")
    
    # Thresholds:
    # 1. Candidate must improve semantic similarity by at least 0.02 (noise margin).
    # 2. Candidate must NOT regress on keyword (factual) matching.
    sim_passed = cand_sim >= (prod_sim + 0.02)
    kw_passed  = cand_kw >= prod_kw
    
    if sim_passed and kw_passed:
        print("✅ GATE PASSED: Candidate beat production by >0.02 Sim and didn't regress on keywords.")
        print("Promoting candidate to production...")
        # The candidate is already at PROD_DIR. Just delete the backup.
        if TEMP_BACKUP_DIR.exists():
            shutil.rmtree(TEMP_BACKUP_DIR)
        update_live_log(f"✅ Eval Gate PASSED. Promoted to Prod. (Sim {cand_sim:.3f}, KW {cand_kw:.0%})")
    else:
        print("❌ GATE FAILED: Candidate degraded performance or failed to beat the margin.")
        print("Rolling back to previous production model...")
        # Move candidate back
        if PROD_DIR.exists():
            PROD_DIR.rename(CANDIDATE_DIR)
        # Restore backup
        if TEMP_BACKUP_DIR.exists():
            TEMP_BACKUP_DIR.rename(PROD_DIR)
        update_live_log(f"❌ Eval Gate FAILED. Rolled back. (Sim {cand_sim:.3f}, KW {cand_kw:.0%})")

    print("="*60)

    
if __name__ == "__main__":
    main()
