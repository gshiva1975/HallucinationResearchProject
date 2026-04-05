# banana_service/llm.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLlamaLLM:
    def __init__(
        self,
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=None
    ):
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"[LLM] Using device: {self.device}")

        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Fix 1: set pad_token — TinyLlama has no pad token by default
        # this is what triggers the torch.isin call that crashes on MPS
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)

        # Fix 2: mirror pad_token_id on the model config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = 60, **kwargs):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                # Fix 3: pass eos explicitly so transformers doesn't
                # call torch.isin to figure it out at runtime on MPS
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result[len(prompt):].strip()
