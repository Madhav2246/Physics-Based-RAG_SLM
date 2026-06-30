"""
req.py — one-shot Kaggle setup + preflight check.
Run FIRST:  !python req.py
Installs deps, verifies GPU, corpus, anchors. Tells you if ready to sweep.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=False)


def main():
    print("=" * 60)
    print("STEP 1 — install deps (torch already on Kaggle, not reinstalled)")
    sh(f"{sys.executable} -m pip install -q -r {ROOT/'requirements.txt'}")

    print("\n" + "=" * 60)
    print("STEP 2 — environment")
    try:
        import torch
        print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
              "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
        if not torch.cuda.is_available():
            print("  [WARN] No GPU. Set Kaggle Accelerator = GPU P100.")
    except Exception as e:
        print("  [FAIL] torch import:", e)
    try:
        import sentence_transformers as st
        print("sentence-transformers:", st.__version__)
    except Exception as e:
        print("  [FAIL] sentence-transformers:", e)

    print("\n" + "=" * 60)
    print("STEP 3 — data files")
    corpus = list((ROOT / "data" / "corpus").glob("*.txt"))
    anchors = ROOT / "data" / "evaluation" / "gold_anchors.json"
    print(f"corpus .txt files : {len(corpus)}  {'[OK]' if corpus else '[MISSING]'}")
    if anchors.exists():
        import json
        n = len(json.loads(anchors.read_text(encoding='utf-8')))
        print(f"gold_anchors.json : {n} anchors  [OK]")
    else:
        print("gold_anchors.json : [MISSING]")

    print("\n" + "=" * 60)
    ok = corpus and anchors.exists()
    print("READY ✅  Run:  !python sweep_retrieval_gpu.py --full --rerank"
          if ok else "NOT READY ❌ — see [MISSING] above")
    print("(faster subset:  !python sweep_retrieval_gpu.py)")


if __name__ == "__main__":
    main()
