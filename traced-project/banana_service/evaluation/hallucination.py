# banana_service/evaluation/hallucination.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from logger import setup_logger

logger = setup_logger("HallucinationEval")


class HallucinationEvaluator:

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self.embedder  = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info(f"HallucinationEvaluator ready  threshold={threshold}")

    def split_sentences(self, text: str) -> list:
        return [s.strip() for s in text.split(".") if s.strip()]

    def evaluate(self, answer: str, retrieved_docs: list) -> dict:
        logger.info(f"Evaluating answer  len={len(answer)}  docs={len(retrieved_docs)}")

        if not retrieved_docs:
            logger.warning("No retrieved docs — hallucination_rate=1.0")
            return {
                "hallucination_rate":  1.0,
                "faithfulness_score":  0.0,
                "unsupported_sentences": self.split_sentences(answer),
            }

        sentences = self.split_sentences(answer)
        logger.info(f"  Answer split into {len(sentences)} sentence(s)")

        doc_embeddings = self.embedder.encode(retrieved_docs)
        unsupported    = []

        for i, sentence in enumerate(sentences):
            sent_emb = self.embedder.encode([sentence])
            sims     = cosine_similarity(sent_emb, doc_embeddings)
            max_sim  = float(np.max(sims))
            supported = max_sim >= self.threshold
            logger.info(f"  Sentence [{i}]  max_sim={max_sim:.4f}  "
                        f"supported={'✓' if supported else '✗'}  "
                        f"text={sentence[:80]!r}")
            if not supported:
                unsupported.append(sentence)

        hallucination_rate = len(unsupported) / max(1, len(sentences))
        faithfulness_score = 1 - hallucination_rate

        logger.info(f"  Result  hallucination={hallucination_rate:.3f}  "
                    f"faithfulness={faithfulness_score:.3f}  "
                    f"unsupported={len(unsupported)}/{len(sentences)}")
        return {
            "hallucination_rate":    hallucination_rate,
            "faithfulness_score":    faithfulness_score,
            "unsupported_sentences": unsupported,
        }
