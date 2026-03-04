
import faiss
import numpy as np
from logger import setup_logger

logger = setup_logger("VectorStore")

class VectorStore:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.docs = []

    def add(self, vectors, docs):
        logger.info(f"Adding {len(docs)} docs to FAISS")
        self.index.add(np.array(vectors).astype("float32"))
        self.docs.extend(docs)

    def search1(self, query_vec, k=3):
        logger.info("Running RAG retrieval")
        D, I = self.index.search(
            np.array([query_vec]).astype("float32"), k
        )
        return [self.docs[i] for i in I[0]]

    def search(self, query_vec, k=3):
        logger.info("Running RAG retrieval")
    
        if len(self.docs) == 0:
            logger.warning("VectorStore empty — no docs to retrieve")
            return []
    
        D, I = self.index.search(
            np.array([query_vec]).astype("float32"), k
        )
    
        results = []
        for i in I[0]:
            if 0 <= i < len(self.docs):
                results.append(self.docs[i])
    
        return results
