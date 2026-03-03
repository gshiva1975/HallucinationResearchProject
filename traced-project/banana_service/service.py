from banana_service.config import settings


class BananaService:

    def __init__(
        self,
        llm,
        store=None,
        embed=None,
        researcher_agent=None
    ):
        self.mode = settings.EXPERIMENT_MODE
        self.llm = llm

        if self.mode == "BASELINE":

            # Import ONLY baseline components
            from banana_service.baseline_model import BaselineFinancialModel

            self.engine = BaselineFinancialModel(llm)

        elif self.mode == "OPTIMIZED":

            # Import optimized pipeline ONLY when needed
            from banana_service.optimized_pipeline import OptimizedBananaPipeline

            if store is None or embed is None or researcher_agent is None:
                raise ValueError(
                    "OPTIMIZED mode requires store, embed, and researcher_agent."
                )

            self.engine = OptimizedBananaPipeline(
                llm=llm,
                store=store,
                embed=embed,
                researcher_agent=researcher_agent
            )

        else:
            raise ValueError(f"Unsupported EXPERIMENT_MODE: {self.mode}")

    def analyze(self, query: str):
        return self.engine.analyze(query)
