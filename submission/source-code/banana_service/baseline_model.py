# banana_service/baseline_model.py

class BaselineFinancialModel:

    def __init__(self, llm):
        self.llm = llm

    def analyze(self, query: str):

        prompt = f"""
You are a financial analyst.

Answer the following financial question with detailed reasoning.

Question:
{query}

Provide:
- Key figures
- Supporting reasoning
- Final summary
"""

        response = self.llm.generate(
            prompt=prompt,
        )

        return {
            "mode": "BASELINE",
            "answer": response,
            "tools_used": [],
            "grounded": False
        }
