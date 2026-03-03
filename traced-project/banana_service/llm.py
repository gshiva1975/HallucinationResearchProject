# banana_service/llm.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalLlamaLLM:

    def __init__(
        self,
        #model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        #model_name="mistralai/Mistral-7B-Instruct-v0.2",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=None
    ):
        #self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = 60, **kwargs):
    
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,        # deterministic
                num_beams=1,            # greedy
                use_cache=True,
            )
    
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result[len(prompt):].strip()
