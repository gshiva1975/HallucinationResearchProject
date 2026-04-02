# banana_service/config.py

import os

class Settings:
    def __init__(self):
        self.EXPERIMENT_MODE = os.getenv("EXPERIMENT_MODE", "OPTIMIZED")

        if self.EXPERIMENT_MODE not in ["BASELINE", "OPTIMIZED"]:
            raise ValueError(
                f"Invalid EXPERIMENT_MODE: {self.EXPERIMENT_MODE}. "
                "Must be BASELINE or OPTIMIZED."
            )

        self.MCP_MARKET_URL = os.getenv("MCP_MARKET_URL")
        self.MCP_SEC_URL = os.getenv("MCP_SEC_URL")
        self.MCP_SOCIAL_URL = os.getenv("MCP_SOCIAL_URL")

settings = Settings()
