"""
train_from_feedback.py
──────────────────────
Fine-tunes the SLM on HITL corrections using a 70/30 correction-to-base-data
mix to prevent catastrophic forgetting.

Key design decisions:
 • Uses HuggingFaceH4/no_robots as the base dataset — human-written, higher
   quality than Alpaca, and closer in style to Qwen-Instruct outputs.
 • Saves the adapter to models/candidate_slm (staging), NOT models/finetuned_slm.
   Run evaluate_candidate.py afterward to gate promotion.
 • 3 epochs max at lr=1e-4 for ~65 examples: enough to learn, not enough to
   memorise.
"""

import argparse
import json
import random
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# ── Live progress callback ─────────────────────────────────────────────────────
class LiveProgressCallback(TrainerCallback):
    def __init__(self, live_file: Path, total_steps: int):
        self.live_file = live_file
        self.total_steps = total_steps
        self.log_history: list[str] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss")
        lr   = logs.get("learning_rate")
        if loss is not None:
            msg = f"Step {step}/{self.total_steps} | Loss: {loss:.4f} | LR: {lr:.2e}"
            self.log_history.append(msg)
            self.live_file.write_text(
                json.dumps({
                    "status":     "training",
                    "step":       step,
                    "max_steps":  self.total_steps,
                    "loss":       loss,
                    "log":        self.log_history,
                }, indent=2),
                encoding="utf-8",
            )


# ── ChatML formatter ───────────────────────────────────────────────────────────
def to_chatml(prompt: str, response: str) -> str:
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


# ── Base data loader (no_robots — human-written, high quality) ─────────────────
def load_base_examples(n: int, seed: int = 42) -> list[dict]:
    """
    Samples n examples from HuggingFaceH4/no_robots.
    Falls back to an empty list on any download failure so training can still
    run (with a warning) if the machine has no internet access.
    """
    try:
        ds = load_dataset("HuggingFaceH4/no_robots", split="train", trust_remote_code=True)
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        examples = []
        for row in ds:
            # Each row has messages: [{"role": "user", ...}, {"role": "assistant", ...}]
            messages = row.get("messages", [])
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            if user_msg and asst_msg:
                examples.append({"text": to_chatml(user_msg, asst_msg)})
        return examples
    except Exception as exc:
        print(f"[WARN] Could not load no_robots base dataset: {exc}")
        print("[WARN] Training will proceed on corrections only (higher forgetting risk).")
        return []


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HITL SFT with 70/30 base data mixing")
    # Hyperparams tuned for ~65 examples at 0.5B scale
    parser.add_argument("--learning_rate",  type=float, default=1e-4,
                        help="Conservative LR for small dataset (default: 1e-4)")
    parser.add_argument("--lora_rank",      type=int,   default=16)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--num_epochs",     type=int,   default=3,
                        help="Max 3 epochs — avoid memorisation on tiny datasets")
    parser.add_argument("--warmup_ratio",   type=float, default=0.1)
    parser.add_argument("--lr_scheduler",   type=str,   default="cosine")
    parser.add_argument("--base_mix_ratio", type=float, default=0.70,
                        help="Fraction of training data to be base (no_robots) examples")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--model_name",     type=str,   default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir",     type=str,   default="models/candidate_slm",
                        help="Directory to save the adapter")
    args = parser.parse_args()

    random.seed(args.seed)

    project_root  = Path(__file__).resolve().parent.parent
    feedback_file = project_root / "data"   / "feedback" / "hitl_corrections.jsonl"
    live_file     = project_root / "data"   / "feedback" / "live_training.json"
    # ── Staging dir: evaluate_candidate.py gates promotion to finetuned_slm ──
    candidate_dir = project_root / args.output_dir

    if not feedback_file.exists():
        print("[FAIL] No feedback file found at", feedback_file)
        return

    # ── 1. Load corrections ────────────────────────────────────────────────────
    corrections: list[dict] = []
    with open(feedback_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("feedback_type") == "correction":
                q   = entry.get("question", "").strip()
                ans = entry.get("correct_response", "").strip()
                if q and ans:
                    corrections.append({"text": to_chatml(q, ans)})

    if not corrections:
        print("[FAIL] No correction-type feedback pairs found in", feedback_file)
        return

    n_corrections = len(corrections)
    if n_corrections < 5:
        print(f"[FAIL] Only {n_corrections} correction(s) found. At least 5 are required to prevent catastrophic forgetting. Collect more feedback first.")
        return
        
    print(f"[OK]  Loaded {n_corrections} HITL correction(s).")

    # ── 2. 70/30 base data mixing ──────────────────────────────────────────────
    # If we have N corrections and want them to be 30% of training data,
    # we need M = N * (base_ratio / correction_ratio) base examples.
    correction_ratio = 1.0 - args.base_mix_ratio
    n_base_needed    = int(n_corrections * (args.base_mix_ratio / correction_ratio))
    print(f"[OK]  Fetching {n_base_needed} base examples from no_robots "
          f"({int(args.base_mix_ratio*100)}/{int(correction_ratio*100)} mix)…")

    base_examples = load_base_examples(n_base_needed, seed=args.seed)
    n_base_actual = len(base_examples)

    if n_base_actual == 0:
        print("[WARN] No base examples loaded — proceeding on corrections only.")
    else:
        print(f"[OK]  Got {n_base_actual} base examples.")

    mixed = corrections + base_examples
    random.shuffle(mixed)
    n_total = len(mixed)
    print(f"[OK]  Total training examples: {n_total} "
          f"({n_corrections} corrections + {n_base_actual} base)")

    dataset  = Dataset.from_list(mixed)
    model_id = args.model_name

    # ── 3. Model + tokenizer ───────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token        = tokenizer.eos_token
    tokenizer.padding_side     = "right"
    tokenizer.model_max_length = 512   # slightly larger than before for no_robots

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # ── 4. LoRA config ─────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r             = args.lora_rank,
        lora_alpha    = args.lora_rank * 2,   # alpha = 2× rank
        lora_dropout  = 0.05,
        bias          = "none",
        task_type     = "CAUSAL_LM",
        target_modules= ["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    # ── 5. SFT config ──────────────────────────────────────────────────────────
    # gradient_accumulation_steps = 2 → effective batch = batch_size × 2 = 8
    sft_config = SFTConfig(
        output_dir                  = str(project_root / "logs" / "hitl_outputs"),
        dataset_text_field          = "text",
        packing                     = False,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = 2,
        optim                       = "adamw_torch",
        num_train_epochs            = args.num_epochs,   # epoch-based, not step-based
        save_strategy               = "no",              # we save manually below
        logging_steps               = 5,
        learning_rate               = args.learning_rate,
        gradient_checkpointing      = True,
        fp16                        = True,
        max_grad_norm               = 0.3,
        warmup_ratio                = args.warmup_ratio,
        lr_scheduler_type           = args.lr_scheduler,
        seed                        = args.seed,
    )

    # Estimate total steps for the live progress bar
    steps_per_epoch = max(1, n_total // (args.batch_size * 2))
    total_steps     = steps_per_epoch * args.num_epochs

    # ── 6. Live log init ───────────────────────────────────────────────────────
    live_file.parent.mkdir(parents=True, exist_ok=True)
    live_file.write_text(json.dumps({
        "status":    "training",
        "step":      0,
        "max_steps": total_steps,
        "loss":      None,
        "log": [
            f"[TinySLM] {n_corrections} corrections + {n_base_actual} no_robots base examples = {n_total} total",
            f"[TinySLM] 70/30 mix · 3 epochs · LR={args.learning_rate:.1e} · LoRA r={args.lora_rank}",
            f"[TinySLM] Adapter will be staged to models/candidate_slm (not yet promoted)",
        ],
    }, indent=2), encoding="utf-8")

    callback = LiveProgressCallback(live_file, total_steps)

    # ── 7. Train ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model            = model,
        train_dataset    = dataset,
        peft_config      = peft_config,
        processing_class = tokenizer,
        args             = sft_config,
        callbacks        = [callback],
    )
    trainer.train()

    # ── 8. Save to STAGING dir, not production ─────────────────────────────────
    candidate_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(candidate_dir))
    tokenizer.save_pretrained(str(candidate_dir))
    print(f"\n[OK]  Candidate adapter saved to: {candidate_dir}")
    print("[INFO] Run evaluate_candidate.py to gate promotion to models/finetuned_slm")

    # ── 9. Final live log ──────────────────────────────────────────────────────
    try:
        final = json.loads(live_file.read_text(encoding="utf-8"))
    except Exception:
        final = {}
    final["status"] = "awaiting_eval"
    final.setdefault("log", []).append(
        "✅ Training complete — adapter staged. Run evaluate_candidate.py to promote."
    )
    live_file.write_text(json.dumps(final, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
