# banana_service/core/embedding_model.py
import time
import torch
from transformers import AutoTokenizer, AutoModel
from logger import setup_logger

logger = setup_logger("EmbeddingModel")


class EmbeddingModel:

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {name}")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model     = AutoModel.from_pretrained(name)
        logger.info("Embedding model ready ✓")

    def encode(self, text: str):
        t0     = time.perf_counter()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        vec     = outputs.last_hidden_state.mean(dim=1)[0].numpy()
        elapsed = round(time.perf_counter() - t0, 4)
        logger.debug(f"encode()  elapsed={elapsed}s  text={text[:60]!r}")
        return vec
