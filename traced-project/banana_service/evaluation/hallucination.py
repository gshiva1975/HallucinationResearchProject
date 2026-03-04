import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class HallucinationEvaluator:

    def __init__(self, threshold=0.65):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

    def split_sentences(self, text):
        return [s.strip() for s in text.split(".") if s.strip()]

    def evaluate(self, answer, retrieved_docs):

        if not retrieved_docs:
            return {
                "hallucination_rate": 1.0,
                "faithfulness_score": 0.0,
                "unsupported_sentences": self.split_sentences(answer)
            }

        answer_sentences = self.split_sentences(answer)

        doc_embeddings = self.embedder.encode(retrieved_docs)
        unsupported = []

        for sentence in answer_sentences:
            sent_embedding = self.embedder.encode([sentence])
            sims = cosine_similarity(sent_embedding, doc_embeddings)
            max_sim = np.max(sims)

            if max_sim < self.threshold:
                unsupported.append(sentence)

        hallucination_rate = len(unsupported) / max(1, len(answer_sentences))

        return {
            "hallucination_rate": hallucination_rate,
            "faithfulness_score": 1 - hallucination_rate,
            "unsupported_sentences": unsupported
        }
