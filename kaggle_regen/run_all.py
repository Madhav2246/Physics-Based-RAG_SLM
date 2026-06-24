"""
run_all.py — regenerate all evaluation outputs on the NEW bge-large index.
==========================================================================
The retrieval index changed (MiniLM/512 -> bge-large/384, Hit@3 0.27->0.94),
so every SYSTEM-generated answer and every downstream physics score is stale.
This reruns the generation+scoring chain IN ORDER on the new index.

What stays FIXED: data/evaluation/nvidia_golden_qa.jsonl (the questions +
ground-truth answers — the exam). We only regenerate the SYSTEM's responses.

Chain (each step depends on the previous writing answers_dump / scores):
  1. stage1_physics_new.py   -> regenerates answers_dump.jsonl + Stage 1 scores
  2. stage2_generation.py    -> Stage 2 (reads answers_dump)
  3. stage4_ablation.py      -> Stage 4
  4. stage4b_validator_test.py -> Stage 4b (generates its own)
  5. stage6_significance.py  -> significance tests

RUN ON KAGGLE (GPU P100, Internet ON):
  %cd /kaggle/working/kaggle_regen/backend_new
  !pip install -q -r requirements.txt
  !python ../run_all.py                  # full chain
  !python ../run_all.py --only 1         # just stage 1 (regen answers_dump)
  !python ../run_all.py --smoke          # 5 questions per stage, fast sanity

Outputs land in backend_new/data/evaluation/. Download that folder when done.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent / "backend_new"

# (label, script, args). stage1 is the keystone (writes answers_dump).
STEPS = [
    ("Stage 1 (regen answers_dump + physics scores)", "stage1_physics_new.py", ["--n", "100", "--samples", "3"]),
    ("Stage 2 (generation quality)",                  "stage2_generation.py",  []),
    ("Stage 4 (ablation)",                            "stage4_ablation.py",    []),
    # Stage 4b: validator-vs-diversity sweep, HARD only, n = 1 3 5 7 9 11 13 15 17.
    # OOM-safe: generation is num_return_sequences=1 looped n times, so n=17 uses
    # the SAME GPU memory as n=1 (only time scales, not memory).
    ("Stage 4b (validator sweep, hard, n=1..17)",     "stage4b_validator_test_tempsweep.py",
     ["--samples", "1", "3", "5", "7", "9", "11", "13", "15", "17", "--max_questions", "20"]),
    ("Stage 6 (significance)",                        "stage6_significance.py", []),
]


EVAL = BACKEND / "data" / "evaluation"
STAGE1_DIR = EVAL / "stage1_new_separate_eval"


def run(script, extra):
    cmd = [sys.executable, str(BACKEND / "scripts" / script)] + extra
    print(f"\n{'='*72}\n$ {' '.join(cmd)}\n{'='*72}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, cwd=str(BACKEND))
    print(f"[{'OK' if r.returncode==0 else 'FAIL rc='+str(r.returncode)}] "
          f"{script}  ({time.time()-t:.0f}s)", flush=True)
    return r.returncode


def promote_stage1_outputs():
    """Stage 1 writes into stage1_new_separate_eval/, but stages 2/4/6 read from
    data/evaluation/. Copy the two shared files up so the chain connects."""
    import shutil
    for name in ("answers_dump.jsonl", "stage1_new.json"):
        src = STAGE1_DIR / name
        if src.exists():
            shutil.copy2(src, EVAL / name)
            print(f"  promoted {name} -> data/evaluation/", flush=True)
        else:
            print(f"  [WARN] stage1 output missing: {src}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, help="run only step N (1-5)")
    ap.add_argument("--smoke", action="store_true", help="5 questions per stage")
    ap.add_argument("--from_step", type=int, default=1, help="start at step N")
    args = ap.parse_args()

    steps = list(enumerate(STEPS, 1))
    if args.only:
        steps = [s for s in steps if s[0] == args.only]
    else:
        steps = [s for s in steps if s[0] >= args.from_step]

    for idx, (label, script, extra) in steps:
        if args.smoke:
            # shrink question count where the script supports it
            extra = list(extra)
            # --n 100 -> --n 5
            if "--n" in extra:
                extra[extra.index("--n") + 1] = "5"
            # --max_questions 20 -> --max_questions 5
            if "--max_questions" in extra:
                extra[extra.index("--max_questions") + 1] = "5"
            elif "--n" not in extra:
                extra = extra + ["--max_questions", "5"]
        print(f"\n###### STEP {idx}: {label} ######")
        rc = run(script, extra)
        if idx == 1 and rc == 0:
            promote_stage1_outputs()   # connect stage1 outputs to stages 2/4/6
        if rc != 0 and not args.only:
            print(f"\n[STOP] Step {idx} failed — fix before continuing "
                  f"(resume with: python run_all.py --from_step {idx})")
            sys.exit(rc)

    print("\nAll requested steps done. Download backend_new/data/evaluation/")


if __name__ == "__main__":
    main()
