
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from logger import setup_logger

logger = setup_logger("FinBERT")

class FinancialSentimentModel:
    def __init__(self, name):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.labels = ["positive","negative","neutral"]

    def predict(self, text):
        logger.info("Running FinBERT inference")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return {
            "label": self.labels[pred.item()],
            "confidence": round(conf.item(),4)
        }
