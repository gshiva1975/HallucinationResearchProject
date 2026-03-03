# banana_service/llm.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalLlamaLLM:

    def __init__(
        self,
        # distilgpt2 = 82M params, fast on CPU (~1-2s/query)
        # alternatives:
        #   "facebook/opt-125m"          — 125M, slightly better quality
        #   "TinyLlama/TinyLlama-1.1B-Chat-v1.0" — original, slow on CPU
        model_name="distilgpt2",
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"  Loading {model_name} on {self.device}…", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # distilgpt2 has no pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.eval()
        print(f"  {model_name} ready.", flush=True)

    def generate(self, prompt: str, max_tokens: int = 80, **kwargs) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,         # keep input short so generation is fast
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,    # greedy — deterministic and faster
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Return only the newly generated tokens (strip the prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
