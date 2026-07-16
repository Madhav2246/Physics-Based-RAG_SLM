import pathlib
_original_read_text = pathlib.Path.read_text
def _utf8_read_text(self, encoding=None, errors=None):
    if encoding is None:
        encoding = 'utf-8'
    return _original_read_text(self, encoding=encoding, errors=errors)
pathlib.Path.read_text = _utf8_read_text

import builtins
_original_open = builtins.open
def _utf8_open(*args, **kwargs):
    mode = kwargs.get('mode', args[1] if len(args) > 1 else 'r')
    if 'b' not in mode and 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return _original_open(*args, **kwargs)
builtins.open = _utf8_open

import os
os.environ["HF_HOME"] = "d:/S6/NLP/Physics_Based_RAG_SLM/hf_cache"
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    data_path = "data/processed/train_dataset.jsonl"
    output_dir = os.path.join("models", "finetuned_slm")   # matches cfg.ADAPTER_PATH

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}. Please run prepare_data.py first.")
        return

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=data_path, split="train")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 256

    print("Hard-truncating text dataset to 128 tokens to prevent cross-entropy OOM...")
    def truncate_text(example):
        tokens = tokenizer.encode(example["text"], truncation=True, max_length=128)
        example["text"] = tokenizer.decode(tokens, skip_special_tokens=True)
        return example
    dataset = dataset.map(truncate_text)

    print("Loading base model in FP16 directly to GPU (bypassing bitsandbytes & accelerate warmup)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    print("Setting up LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    print("Configuring training arguments...")
    sft_config = SFTConfig(
        output_dir="./logs/trainer_outputs",
        dataset_text_field="text",
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        optim="adamw_torch",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        gradient_checkpointing=True,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=200, # Set to a low number for quick fine-tuning/demonstration
        warmup_steps=10,
        lr_scheduler_type="cosine",
    )

    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving fine-tuned adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")

if __name__ == "__main__":
    main()
