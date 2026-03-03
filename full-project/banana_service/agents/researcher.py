# banana_service/agents/researcher.py

import logging
from banana_service.ingestion.mcp_client import MCPClient
from banana_service.config import settings
import re

logger = logging.getLogger("Researcher")


class ResearcherAgent:

    def __init__(self, store, embed):
        self.store = store
        self.embed = embed

        # MCP clients — only instantiate if URLs are configured
        self.sec    = MCPClient(settings.MCP_SEC_URL)    if settings.MCP_SEC_URL    else None
        self.market = MCPClient(settings.MCP_MARKET_URL) if settings.MCP_MARKET_URL else None
        self.social = MCPClient(settings.MCP_SOCIAL_URL) if settings.MCP_SOCIAL_URL else None

    def extract_ticker(self, query: str):
        """Extract an all-caps 2–5 letter ticker from the query."""
        match = re.search(r"\b[A-Z]{2,5}\b", query)
        return match.group(0) if match else None

    def run(self, state: dict) -> dict:

        logger.info("Researcher retrieving documents")

        ticker = self.extract_ticker(state["query"])
        docs = []

        if ticker:
            logger.info(f"Fetching MCP data for ticker: {ticker}")

            if self.sec:
                try:
                    results = self.sec.call_tool("fetch_sec_filings", {"ticker": ticker})
                    docs += results if isinstance(results, list) else [results]
                except Exception as e:
                    logger.warning(f"SEC MCP failed: {e}")

            if self.market:
                try:
                    results = self.market.call_tool("fetch_market_data", {"ticker": ticker})
                    docs += results if isinstance(results, list) else [results]
                except Exception as e:
                    logger.warning(f"Market MCP failed: {e}")

            if self.social:
                try:
                    results = self.social.call_tool("fetch_social_sentiment", {"ticker": ticker})
                    docs += results if isinstance(results, list) else [results]
                except Exception as e:
                    logger.warning(f"Social MCP failed: {e}")

            # Embed fresh documents and add to vector store
            if docs:
                vectors = [self.embed.encode(d) for d in docs]
                self.store.add(vectors, docs)

        # Semantic retrieval against stored vectors
        vec = self.embed.encode(state["query"])
        retrieved_docs = self.store.search(vec)

        return {**state, "docs": retrieved_docs}
