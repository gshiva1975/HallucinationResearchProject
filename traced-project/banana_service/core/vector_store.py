import numpy as np
import logging

logger = logging.getLogger("VectorStore")


class VectorStore:

    def __init__(self, dim=384):
        self.dim  = dim
        self.vecs = []
        self.docs = []

    def add(self, vectors, docs):
        logger.info(f"Adding {len(docs)} docs to VectorStore")
        for v, d in zip(vectors, docs):
            self.vecs.append(np.array(v, dtype="float32"))
            self.docs.append(d)

    def search(self, query_vec, k=3):
        logger.info("Running similarity search")

        if not self.vecs:
            logger.warning("VectorStore empty — no docs to retrieve")
            return []

        q = np.array(query_vec, dtype="float32")
        matrix  = np.stack(self.vecs)
        q_norm  = q / (np.linalg.norm(q) + 1e-10)
        m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        sims    = matrix / m_norms @ q_norm

        top_k = min(k, len(self.docs))
        top_indices = np.argsort(sims)[::-1][:top_k]

        return [self.docs[i] for i in top_indices]
