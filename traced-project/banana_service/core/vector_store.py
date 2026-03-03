# banana_service/core/vector_store.py
import numpy as np
from logger import setup_logger

logger = setup_logger("VectorStore")


class VectorStore:

    def __init__(self, dim: int = 384):
        self.dim  = dim
        self.vecs = []
        self.docs = []
        logger.info(f"VectorStore initialised  dim={dim}")

    def add(self, vectors, docs):
        before = len(self.docs)
        for v, d in zip(vectors, docs):
            self.vecs.append(np.array(v, dtype="float32"))
            self.docs.append(d)
        logger.info(f"add()  +{len(docs)} docs  total={len(self.docs)}  (was {before})")

    def search(self, query_vec, k: int = 3) -> list:
        if not self.vecs:
            logger.warning("search()  VectorStore is empty — returning []")
            return []

        q       = np.array(query_vec, dtype="float32")
        matrix  = np.stack(self.vecs)
        q_norm  = q / (np.linalg.norm(q) + 1e-10)
        m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        sims    = matrix / m_norms @ q_norm

        top_k   = min(k, len(self.docs))
        indices = np.argsort(sims)[::-1][:top_k]

        results = [self.docs[i] for i in indices]
        scores  = [float(sims[i]) for i in indices]
        logger.info(f"search()  k={k}  total_docs={len(self.docs)}  returned={len(results)}")
        for i, (doc, score) in enumerate(zip(results, scores)):
            logger.info(f"  [{i}] sim={score:.4f}  doc={doc[:100]!r}")
        return results
