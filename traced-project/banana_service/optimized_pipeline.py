# banana_service/optimized_pipeline.py
from banana_service.evaluation.hallucination import HallucinationEvaluator
from logger import setup_logger, trace_step

logger = setup_logger("OptimizedPipeline")


class OptimizedBananaPipeline:

    def __init__(self, llm, store, embed, researcher_agent):
        self.llm              = llm
        self.store            = store
        self.embed            = embed
        self.researcher_agent = researcher_agent
        self.evaluator        = HallucinationEvaluator()
        logger.info("OptimizedBananaPipeline ready")

    def analyze(self, query: str) -> dict:
        logger.info(f"=== analyze() START  query={query!r}")

        # 1. Retrieve docs via researcher (MCP + vector store)
        with trace_step(logger, "researcher.run", query=query[:60]):
            state        = self.researcher_agent.run({"query": query})
            docs         = state.get("docs", [])
        logger.info(f"  Docs retrieved: {len(docs)}")

        context = "\n".join(docs)

        # 2. Build grounded prompt
        prompt = (
            "You are a financial analyst.\n\n"
            "Answer the question strictly using the provided documents.\n"
            "Do NOT repeat the prompt. Do NOT invent facts.\n"
            "If the answer is not in the documents, say: Insufficient data.\n\n"
            f"Documents:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Provide grounded reasoning."
        )
        logger.info(f"  Prompt length: {len(prompt)} chars")

        # 3. Generate answer
        with trace_step(logger, "llm.generate"):
            response = self.llm.generate(prompt)
        logger.info(f"  LLM response: {response[:120]!r}")

        # 4. Evaluate hallucination
        with trace_step(logger, "hallucination.evaluate"):
            metrics = self.evaluator.evaluate(response, docs)
        logger.info(f"  Metrics: {metrics}")

        result = {
            "mode":                  "OPTIMIZED",
            "answer":                response,
            "tools_used":            ["market", "sec", "social"],
            "grounded":              True,
            "hallucination_rate":    metrics["hallucination_rate"],
            "faithfulness_score":    metrics["faithfulness_score"],
            "unsupported_sentences": metrics["unsupported_sentences"],
        }
        logger.info(f"=== analyze() DONE  "
                    f"hallucination={metrics['hallucination_rate']:.3f}  "
                    f"faithfulness={metrics['faithfulness_score']:.3f}")
        return result
