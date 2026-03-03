# banana_service/baseline_model.py
from logger import setup_logger, trace_step

logger = setup_logger("BaselineModel")


class BaselineFinancialModel:

    def __init__(self, llm):
        self.llm = llm
        logger.info("BaselineFinancialModel ready (no retrieval)")

    def analyze(self, query: str) -> dict:
        logger.info(f"=== analyze() START  query={query!r}")

        prompt = (
            "You are a financial analyst.\n\n"
            "Answer the following financial question with detailed reasoning.\n\n"
            f"Question:\n{query}\n\n"
            "Provide:\n- Key figures\n- Supporting reasoning\n- Final summary"
        )
        logger.info(f"  Prompt length: {len(prompt)} chars  (no retrieved context)")

        with trace_step(logger, "llm.generate"):
            response = self.llm.generate(prompt)
        logger.info(f"  LLM response: {response[:120]!r}")
        logger.warning("  ⚠ BASELINE: answer not grounded in any retrieved documents")

        return {
            "mode":       "BASELINE",
            "answer":     response,
            "tools_used": [],
            "grounded":   False,
        }
