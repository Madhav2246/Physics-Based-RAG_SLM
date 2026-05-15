import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

class TinySLM:

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        ).to("cuda")

        adapter_path = "models/finetuned_slm"
        if os.path.exists(adapter_path):
            print(f"Loading LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.config.use_cache = False

    def generate_multiple(self, prompt, n_samples=3, max_tokens=128):
        messages = [
            {"role": "system", "content": "You are a semiconductor device physics assistant."},
            {"role": "user", "content": prompt}
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=n_samples
        )

        return [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]