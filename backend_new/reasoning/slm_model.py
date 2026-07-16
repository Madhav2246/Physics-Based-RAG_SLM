from __future__ import annotations
import os
import torch
torch.set_num_threads(4)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import utils.config as cfg


class TinySLM:
    """
    Thin wrapper around Qwen 2.5-0.5B-Instruct.

    Key fixes applied:
    - Auto-detects CUDA vs CPU and selects appropriate dtype (fp16 on GPU, fp32 on CPU).
    - Decodes ONLY newly generated tokens — strips the input prompt from output.
    - Loads LoRA adapter from the correct path (cfg.ADAPTER_PATH).
    - MAX_NEW_TOKENS, N_SAMPLES, TEMPERATURE, TOP_P all driven by config.py.
    """

    def __init__(self, model_name: str = None, adapter_path: str = None):
        model_name = model_name or cfg.MODEL_NAME

        # -- Device + dtype ----------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # float16 is fast on CUDA; float32 required for CPU (many ops unsupported in fp16)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        print(f"[TinySLM] device={self.device} | dtype={self.dtype}")

        # -- Tokenizer ---------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # -- Base model --------------------------------------------------------
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        ).to(self.device)

        # -- LoRA adapter (optional) -------------------------------------------
        adapter_path = adapter_path or cfg.ADAPTER_PATH
        if adapter_path and os.path.exists(adapter_path):
            print(f"[TinySLM] Loading LoRA adapter from '{adapter_path}'...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        else:
            print(f"[TinySLM] No adapter found at '{adapter_path}' — using base model.")

        self.model.config.use_cache = True

    def generate_multiple(self, prompt: str, n_samples: int = None,
                          max_tokens: int = None, seed: int = None) -> list[str]:
        """
        Generate n_samples responses for the given prompt.

        Returns only the newly generated text — the input prompt is stripped
        by slicing the output tensor from position input_length onward.

        Reproducibility (Tier 1): if `seed` is given (or cfg.SEED is set), the
        RNG is reseeded right before sampling so identical calls produce
        identical outputs. This is what stops the eval score bouncing between
        otherwise-identical runs. Pass seed=None AND set cfg.SEED=None to
        restore fully stochastic behaviour.
        """
        n_samples = n_samples if n_samples is not None else cfg.N_SAMPLES
        max_tokens = max_tokens if max_tokens is not None else cfg.MAX_NEW_TOKENS

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a semiconductor physics assistant. "
                    "Explain the physical meaning of each symbol in an equation using a bulleted list. "
                    "Use PLAIN TEXT ONLY. Do NOT use LaTeX, \\[, \\(, $, or \\frac. "
                    "Example output:\n"
                    "- Id = drain current in amperes\n"
                    "- Cox = oxide capacitance per unit area in F/m^2\n"
                    "Base your answer ONLY on the provided Evidence."
                )
            },
            {"role": "user", "content": prompt}
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]   # number of prompt tokens

        # -- Reproducibility: reseed right before sampling ---------------------
        seed = seed if seed is not None else getattr(cfg, "SEED", None)
        if seed is not None:
            from transformers import set_seed
            set_seed(int(seed))

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=cfg.TEMPERATURE,
            top_p=cfg.TOP_P,
            num_return_sequences=n_samples,
        )

        # -- FIX: decode ONLY the newly generated tokens -----------------------
        # Previously: self.tokenizer.decode(output, ...) → included full prompt
        # Now: output[input_length:] → only what the model generated
        # Decode ONLY the newly generated tokens (prompt is already stripped by input_length)
        return [
            self.tokenizer.decode(
                output[input_length:],
                skip_special_tokens=True
            ).strip()
            for output in outputs
        ]