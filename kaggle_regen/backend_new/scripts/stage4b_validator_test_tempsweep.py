"""
stage4b_validator_test_tempsweep.py
-----------------------------------
Multi-n + temperature-sweep variant of the Stage-4b validator test (HARD only).
OOM-safe sequential generation, checkpoint+resume, atomic writes.

The validator's selection gap (sys_best - sys_rand) only has meaning when samples
are DIVERSE, so temperature is a CLI arg (default 0.9, vs 0.3 deployment temp).
Sweep it to show the gap GROWS with diversity:

  python scripts/stage4b_validator_test_tempsweep.py --samples 7 --temperature 0.3
  python scripts/stage4b_validator_test_tempsweep.py --samples 7 --temperature 0.6
  python scripts/stage4b_validator_test_tempsweep.py --samples 7 --temperature 0.9

NOTE: each temperature run overwrites the same per-n files. Rename/move outputs
between runs, or change --max_questions, if you want to keep all three.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import io
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import utils.config as cfg
from physics.new_checker import score_text

OUT_DIR = ROOT / "data" / "evaluation" / "stage4b_validator_hard"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MASTER_JSON = OUT_DIR / "stage4b_validator_hard.json"


def atomic_json_save(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _agg(score_dicts):
    nvr_ok = nvr_eval = 0
    dcr_ok = dcr_eval = 0
    for s in score_dicts:
        if not s["parseable"]:
            continue
        nm = s["num_msg"]
        if "[OK]" in nm:
            nvr_ok += 1
            nvr_eval += 1
        elif "Unresolved" not in nm and "skipped" not in nm.lower():
            nvr_eval += 1
        if s["coverage_frac"] >= 0.999:
            dcr_eval += 1
            if s["dimensional"]:
                dcr_ok += 1
    return {
        "avg_score": round(_avg([s["total"] for s in score_dicts]), 3),
        "parseable": round(100 * _avg([1 if s["parseable"] else 0 for s in score_dicts]), 1),
        "coverage": round(100 * _avg([s["coverage_frac"] for s in score_dicts]), 1),
        "nvr_cond": round(100 * nvr_ok / nvr_eval, 1) if nvr_eval else None,
        "dcr_cond": round(100 * dcr_ok / dcr_eval, 1) if dcr_eval else None,
    }


def compute_summary(agg_buckets, sampling=None):
    summary = {k: _agg(v) for k, v in agg_buckets.items()}
    summary["_analysis"] = {
        "validator_gap_SYS": round(summary["sys_best"]["avg_score"] - summary["sys_rand"]["avg_score"], 3),
        "validator_gap_RAW": round(summary["raw_best"]["avg_score"] - summary["raw_first"]["avg_score"], 3),
        "bestofN_gain": round(summary["sys_best"]["avg_score"] - summary["sys_first"]["avg_score"], 3),
    }
    if sampling is not None:
        summary["_sampling"] = sampling
    return summary


def _make_sys(corpus_eq, sample):
    return (f"Equation: {corpus_eq}\n\n{sample}" if corpus_eq
            else f"Equation: NOT FOUND IN CORPUS\n\n{sample}")


def load_checkpoint(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def parse_n_from_filename(path: Path):
    m = re.search(r"_n(\d+)\.json$", path.name)
    return int(m.group(1)) if m else None


def rebuild_master_json():
    merged = {}
    for p in sorted(OUT_DIR.glob("stage4b_validator_hard_n*.json")):
        if p.name == MASTER_JSON.name:
            continue
        n_val = parse_n_from_filename(p)
        if n_val is None:
            continue
        try:
            merged[f"n_{n_val}"] = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Skipping unreadable file {p.name}: {e}")
    atomic_json_save(MASTER_JSON, merged)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", type=int, default=[7])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_questions", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    sampling = {"temperature": args.temperature, "top_p": args.top_p,
                "top_k": args.top_k, "repetition_penalty": 1.05,
                "max_new_tokens": cfg.MAX_NEW_TOKENS}

    golden = ROOT / "data" / "evaluation" / "nvidia_golden_qa.jsonl"
    rows = []
    with open(golden, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            if x.get("difficulty") == "hard":
                rows.append(x)
    rows = rows[: args.max_questions]
    print(f"Loaded {len(rows)} HARD QA | temp={args.temperature} "
          f"top_p={args.top_p} top_k={args.top_k} | max_tokens={cfg.MAX_NEW_TOKENS}")

    from pipeline.rag_pipeline import RAGPipeline
    from reasoning.prompt_builder import build_prompt
    from transformers import set_seed

    pipe = RAGPipeline()
    slm = pipe.slm
    tokenizer = slm.tokenizer
    model = slm.model
    device = getattr(slm, "device", None) or next(model.parameters()).device

    SYSTEM_PROMPT = getattr(slm, "SYSTEM_PROMPT", None) or (
        "You are a semiconductor physics assistant. "
        "Explain the physical meaning of each symbol in an equation using a bulleted list. "
        "Use PLAIN TEXT ONLY. Do NOT use LaTeX, \\[, \\(, $, or \\frac. "
        "Example output:\n"
        "- Id = drain current in amperes\n"
        "- Cox = oxide capacitance per unit area in F/m^2\n"
        "Base your answer ONLY on the provided Evidence."
    )

    def generate_multiple_safe(prompt, n_samples=1, max_tokens=None, seed=42):
        max_tokens = max_tokens if max_tokens is not None else cfg.MAX_NEW_TOKENS
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]
        model.eval()
        results = []
        for j in range(n_samples):
            set_seed(int(seed) + j)
            output = None
            try:
                with torch.inference_mode():
                    output = model.generate(
                        **inputs, max_new_tokens=max_tokens, do_sample=True,
                        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        num_return_sequences=1, use_cache=True, repetition_penalty=1.05,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id)
                text = tokenizer.decode(output[0, input_length:],
                                        skip_special_tokens=True).strip()
                results.append(text if text else "[EMPTY_GENERATION]")
            finally:
                if output is not None:
                    del output
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        del inputs
        cleanup_memory()
        return results

    pipe.slm.generate_multiple = generate_multiple_safe
    pipe.retriever.dense.load_index()
    pipe.retriever.sparse.build_index_from_docs(pipe.retriever.dense.documents)
    print("Pipeline ready.")
    print("=" * 72)

    t0 = time.time()
    final_results = {}

    for n_samples in args.samples:
        print(f"\nRunning n={n_samples}")
        checkpoint_file = OUT_DIR / f"checkpoint_n_{n_samples}.json"
        dump_file = OUT_DIR / f"answers_dump_hard_n{n_samples}.jsonl"
        n_json = OUT_DIR / f"stage4b_validator_hard_n{n_samples}.json"
        agg_buckets = {k: [] for k in
                       ["sys_best", "sys_first", "sys_rand", "raw_best", "raw_first"]}

        start_idx = 0
        if args.resume:
            ckpt = load_checkpoint(checkpoint_file)
            if ckpt:
                start_idx = ckpt["last_completed"] + 1
                agg_buckets = ckpt["agg_buckets"]
                print(f"Resuming from Q{start_idx + 1}/{len(rows)}")
        else:
            dump_file.write_text("", encoding="utf-8")

        for i in range(start_idx, len(rows)):
            item = rows[i]
            q = item["question"].strip()
            print(f"[n={n_samples}] Q{i + 1}/{len(rows)}")
            try:
                evidence = pipe.retriever.retrieve(q, top_k=cfg.TOP_K)
                _, _, corpus_eq = pipe._find_corpus_equation(evidence)
                prompt = build_prompt(q, evidence, corpus_equation=corpus_eq)
                samples = pipe.slm.generate_multiple(
                    prompt, n_samples=n_samples, seed=(cfg.SEED or 42) + i)

                with open(dump_file, "a", encoding="utf-8") as af:
                    af.write(json.dumps({
                        "n_samples": n_samples, "temperature": args.temperature,
                        "id": item.get("id", f"Q{i + 1}"), "question": q,
                        "corpus_eq": corpus_eq, "samples": samples,
                    }, ensure_ascii=False) + "\n")

                sys_scores = [score_text(_make_sys(corpus_eq, s), "SYS") for s in samples]
                raw_scores = [score_text(s, "RAW") for s in samples]
                rng = random.Random(42 + i)
                agg_buckets["sys_best"].append(max(sys_scores, key=lambda d: d["total"]))
                agg_buckets["sys_first"].append(sys_scores[0])
                agg_buckets["sys_rand"].append(rng.choice(sys_scores))
                agg_buckets["raw_best"].append(max(raw_scores, key=lambda d: d["total"]))
                agg_buckets["raw_first"].append(raw_scores[0])

                atomic_json_save(checkpoint_file, {
                    "n_samples": n_samples, "last_completed": i, "agg_buckets": agg_buckets})
                atomic_json_save(n_json, compute_summary(agg_buckets, sampling))

                del samples, sys_scores, raw_scores
                cleanup_memory()
            except torch.cuda.OutOfMemoryError:
                print("\nCUDA OOM! Resume with --resume")
                cleanup_memory()
                raise

        summary = compute_summary(agg_buckets, sampling)
        final_results[f"n_{n_samples}"] = summary
        atomic_json_save(n_json, summary)
        cleanup_memory()

    merged = rebuild_master_json()
    for k, v in final_results.items():
        merged[k] = v
    atomic_json_save(MASTER_JSON, merged)

    print("\n" + "=" * 72)
    print(f"FINAL SUMMARY  (temperature={args.temperature})")
    print(f"{'n':<6}{'SYS gap':>10}{'RAW gap':>10}{'BoN gain':>12}")
    for n in args.samples:
        src = final_results.get(f"n_{n}") or merged.get(f"n_{n}")
        if not src or "_analysis" not in src:
            print(f"{n:<6}{'(no data)':>32}")
            continue
        x = src["_analysis"]
        print(f"{n:<6}{x['validator_gap_SYS']:>10.3f}"
              f"{x['validator_gap_RAW']:>10.3f}{x['bestofN_gain']:>12.3f}")
    print("=" * 72)
    print(f"Saved -> {MASTER_JSON}  |  Time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
