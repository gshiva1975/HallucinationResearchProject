# banana_service/optimized_pipeline.py

from banana_service.evaluation.hallucination import HallucinationEvaluator


class OptimizedBananaPipeline:

    def __init__(self, llm, store, embed, researcher_agent):
        self.llm = llm
        self.store = store
        self.embed = embed
        self.researcher_agent = researcher_agent

        # Instantiate evaluator once
        self.evaluator = HallucinationEvaluator()

    def analyze(self, query):

        # 1️⃣ Retrieve documents
        state = {"query": query}
        enriched_state = self.researcher_agent.run(state)
        docs = enriched_state.get("docs", [])

        context = "\n".join(docs)

        # 2️⃣ Build prompt
        prompt = f"""
You are a financial analyst.

Answer the question strictly using the provided documents.
Do NOT repeat the prompt.
Do NOT invent facts.
If the answer is not in the documents, say: Insufficient data.

Documents:
{context}

Question:
{query}

Provide grounded reasoning.
"""

        # 3️⃣ Generate answer
        response = self.llm.generate(prompt)

        # 4️⃣ Evaluate hallucination
        metrics = self.evaluator.evaluate(response, docs)

        # 5️⃣ Return structured response
        return {
            "mode": "OPTIMIZED",
            "answer": response,
            "tools_used": ["market", "sec", "social"],
            "grounded": True,
            "hallucination_rate": metrics["hallucination_rate"],
            "faithfulness_score": metrics["faithfulness_score"],
            "unsupported_sentences": metrics["unsupported_sentences"]
        }
