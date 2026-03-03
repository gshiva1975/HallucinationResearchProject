# banana_service/agents/researcher.py
import logging
import re

from banana_service.ingestion.mcp_client import MCPClient
from banana_service.config import settings
from logger import setup_logger, trace_step

logger = setup_logger("Researcher")


class ResearcherAgent:

    def __init__(self, store, embed):
        self.store  = store
        self.embed  = embed
        self.sec    = MCPClient(settings.MCP_SEC_URL)    if settings.MCP_SEC_URL    else None
        self.market = MCPClient(settings.MCP_MARKET_URL) if settings.MCP_MARKET_URL else None
        self.social = MCPClient(settings.MCP_SOCIAL_URL) if settings.MCP_SOCIAL_URL else None
        logger.info(f"ResearcherAgent ready  "
                    f"sec={'✓' if self.sec else '✗'}  "
                    f"market={'✓' if self.market else '✗'}  "
                    f"social={'✓' if self.social else '✗'}")

    def extract_ticker(self, query: str):
        m = re.search(r"\b[A-Z]{2,5}\b", query)
        ticker = m.group(0) if m else None
        logger.info(f"  Ticker extracted: {ticker!r}  from query={query!r}")
        return ticker

    def _call(self, client, tool: str, ticker: str, label: str) -> list:
        if not client:
            logger.warning(f"  {label}: client not configured — skipping")
            return []
        with trace_step(logger, f"mcp/{label}", tool=tool, ticker=ticker):
            try:
                results = client.call_tool(tool, {"ticker": ticker})
                items   = results if isinstance(results, list) else [results]
                logger.info(f"  {label}: got {len(items)} item(s)")
                for i, item in enumerate(items):
                    logger.debug(f"    [{i}] {str(item)[:120]}")
                return items
            except Exception as e:
                logger.error(f"  {label}: call failed — {e}")
                return []

    def run(self, state: dict) -> dict:
        logger.info(f"=== ResearcherAgent.run  query={state['query']!r}")
        docs = []

        ticker = self.extract_ticker(state["query"])
        if ticker:
            docs += self._call(self.sec,    "fetch_sec_filings",    ticker, "SEC")
            docs += self._call(self.market, "fetch_market_data",    ticker, "MARKET")
            docs += self._call(self.social, "fetch_social_sentiment", ticker, "SOCIAL")

            if docs:
                with trace_step(logger, "embed_and_store", n_docs=len(docs)):
                    vectors = [self.embed.encode(d) for d in docs]
                    self.store.add(vectors, docs)
                    logger.info(f"  Embedded and stored {len(docs)} doc(s)")
            else:
                logger.warning("  No docs retrieved from any MCP source")
        else:
            logger.warning("  No ticker found in query — skipping MCP calls")

        with trace_step(logger, "semantic_retrieve", query=state["query"][:60]):
            vec           = self.embed.encode(state["query"])
            retrieved     = self.store.search(vec)
            logger.info(f"  Retrieved {len(retrieved)} doc(s) from vector store")
            for i, d in enumerate(retrieved):
                logger.debug(f"    [{i}] {d[:100]}")

        return {**state, "docs": retrieved}
