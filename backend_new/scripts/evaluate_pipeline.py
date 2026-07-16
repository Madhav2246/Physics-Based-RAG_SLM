import json
import time
import sys
import io
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows (cp1252 chokes on [OK] [WARN] [FAIL] characters)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pathlib
_orig_read_text = pathlib.Path.read_text
def _utf8_read_text(self, encoding=None, errors=None):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors)
pathlib.Path.read_text = _utf8_read_text

import builtins
_orig_open = builtins.open
def _utf8_open(*args, **kwargs):
    mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
    if "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _orig_open(*args, **kwargs)
builtins.open = _utf8_open

import os
os.environ["HF_HOME"] = "d:/S6/NLP/Physics_Based_RAG_SLM/hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.rag_pipeline import RAGPipeline

# Global default (overridden in main)
RESULTS_OUTPUT_PATH = Path("data/evaluation/eval_results.jsonl")

def main(dataset_path: str):
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        print(f"Dataset not found at {dataset_file}")
        return
        
    global RESULTS_OUTPUT_PATH
    RESULTS_OUTPUT_PATH = dataset_file.parent / f"eval_results_{dataset_file.stem}.jsonl"
    LIVE_EVAL_PATH = dataset_file.parent / f"live_evaluation_{dataset_file.stem}.json"

    print("Initializing RAG Pipeline (loading weights)...")
    pipeline = RAGPipeline()
    
    print("Loading vector indexes...")
    pipeline.retriever.dense.load_index()
    pipeline.retriever.sparse.build_index_from_docs(pipeline.retriever.dense.documents)
    
    print(f"Loading dataset from {dataset_file}...")
    dataset = []
    
    if dataset_file.suffix == ".json":
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))

    RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    total = len(dataset)
    high_conf_correct = 0
    low_conf_caught = 0
    
    existing_questions = set()
    if RESULTS_OUTPUT_PATH.exists():
        try:
            with open(RESULTS_OUTPUT_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        existing_questions.add(record["question"])
                        # Recalculate metrics from already processed records
                        if record.get('confidence_score', 0) >= 0.8:
                            high_conf_correct += 1
                        elif record.get('confidence_score', 0) < 0.55:
                            low_conf_caught += 1
        except Exception as e:
            print(f"Error loading existing progress: {e}")
            
    print(f"\nStarting Evaluation over {total} queries (skipping {len(existing_questions)} already done)...\n" + "="*50)
    
    with open(RESULTS_OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        for i, item in enumerate(dataset):
            question = item["question"]
            expected_answer = item["answer"]
            
            if question in existing_questions:
                print(f"[{i+1}/{total}] Q: {question} (Skipped - already evaluated)")
                # Update live evaluation JSON file for progress percentage
                import datetime
                live_data = {
                    "status": "evaluating",
                    "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "overall": {
                        "done": i + 1,
                        "target": total,
                        "percent": round((i + 1) / total * 100, 1)
                    },
                    "metrics": {
                        "high_conf_correct": high_conf_correct,
                        "low_conf_caught": low_conf_caught
                    },
                    "last": {
                        "question": question,
                        "confidence_score": record.get('confidence_score', 0.0) if 'record' in locals() else 0.0,
                        "confidence_label": record.get('confidence_label', 'SKIPPED') if 'record' in locals() else 'SKIPPED'
                    }
                }
                with open(LIVE_EVAL_PATH, "w", encoding="utf-8") as lf:
                    json.dump(live_data, lf, ensure_ascii=False, indent=2)
                continue
            
            print(f"\n[{i+1}/{total}] Q: {question}")
            
            start_time = time.time()
            result = pipeline.answer(question)
            latency = time.time() - start_time
            
            # Record result
            eval_record = {
                "question": question,
                "expected_answer": expected_answer,
                "actual_response": result["response"],
                "semantic_similarity": result["semantic_similarity"],
                "confidence_score": result["confidence_score"],
                "confidence_label": result["confidence_label"],
                "symbolic_validation": result["symbolic_validation"],
                "dimension_validation": result["dimension_validation"],
                "numerical_validation": result["numerical_validation"],
                "latency_sec": latency
            }
            
            out_f.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
            out_f.flush()
            
            print(f"  Confidence: {result['confidence_label']} ({result['confidence_score']})")
            print(f"  Symbolic:   {result['symbolic_validation']}")
            print(f"  Semantic:   {result['semantic_similarity']:.3f}" if result['semantic_similarity'] else "  Semantic:   None")
            
            if result['confidence_score'] >= 0.8:
                high_conf_correct += 1
            elif result['confidence_score'] < 0.55:
                low_conf_caught += 1
                
            # Write to live_evaluation.json
            import datetime
            live_data = {
                "status": "evaluating",
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "overall": {
                    "done": i + 1,
                    "target": total,
                    "percent": round((i + 1) / total * 100, 1)
                },
                "metrics": {
                    "high_conf_correct": high_conf_correct,
                    "low_conf_caught": low_conf_caught
                },
                "last": {
                    "question": question,
                    "confidence_score": result['confidence_score'],
                    "confidence_label": result['confidence_label']
                }
            }
            with open(LIVE_EVAL_PATH, "w", encoding="utf-8") as lf:
                json.dump(live_data, lf, ensure_ascii=False, indent=2)

    # Final update
    live_data["status"] = "complete"
    with open(LIVE_EVAL_PATH, "w", encoding="utf-8") as lf:
        json.dump(live_data, lf, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print("Evaluation Complete!")
    print(f"Results saved to: {RESULTS_OUTPUT_PATH}")
    print(f"High Confidence Answers (Expected good): {high_conf_correct}/{total}")
    print(f"Low Confidence Answers (Caught errors):  {low_conf_caught}/{total}")
    print("Use the eval_results.jsonl file to generate your confusion matrix and P-R curves for the paper.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/evaluation/golden_qa.jsonl")
    args = parser.parse_args()
    main(args.dataset)
