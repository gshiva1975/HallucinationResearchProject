from logger import setup_logger
from banana_service.ingestion.mcp_client import MCPClient
from banana_service.config import settings
import re

logger = setup_logger("Researcher")

class ResearcherAgent:

    def __init__(self, store, embed):
        self.store = store
        self.embed = embed

        # MCP clients
        self.sec = MCPClient(settings.MCP_SEC_URL) if settings.MCP_SEC_URL else None
        self.market = MCPClient(settings.MCP_MARKET_URL) if settings.MCP_MARKET_URL else None
        self.social = MCPClient(settings.MCP_SOCIAL_URL) if settings.MCP_SOCIAL_URL else None

    def extract_ticker(self, query):
        # Very simple ticker extraction (improve later)
        match = re.search(r"\b[A-Z]{2,5}\b", query)
        return match.group(0) if match else None

    def run1(self, state):

        logger.info("Researcher retrieving documents")

        ticker = self.extract_ticker(state["query"])

        if ticker:
            logger.info(f"Fetching MCP data for {ticker}")

            docs = (
                self.sec.call("fetch_sec_filings", {"ticker": ticker}) +
                self.market.call("fetch_market_data", {"ticker": ticker}) +
                self.social.call("fetch_social_sentiment", {"ticker": ticker})
            )

            # Embed fresh documents
            vectors = [self.embed.encode(d) for d in docs]

            # Add to vector store
            self.store.add(vectors, docs)

        # Now perform semantic retrieval
        vec = self.embed.encode(state["query"])
        docs = self.store.search(vec)

        return {**state, "docs": docs}
    def run(self, state):
    
        logger.info("Researcher retrieving documents")
    
        ticker = self.extract_ticker(state["query"])
        docs = []
    
        if ticker:
            logger.info(f"Fetching MCP data for {ticker}")
    
            # Only call MCP services if they exist
            if self.sec:
                try:
                    docs += self.sec.call("fetch_sec_filings", {"ticker": ticker})
                except Exception as e:
                    logger.warning(f"SEC MCP failed: {e}")
    
            if self.market:
                try:
                    docs += self.market.call("fetch_market_data", {"ticker": ticker})
                except Exception as e:
                    logger.warning(f"Market MCP failed: {e}")
    
            if self.social:
                try:
                    #docs += self.social.call("fetch_social_sentiment", {"ticker": ticker})
                    docs += self.social.call_tool( "fetch_social_sentiment", {"ticker": ticker})
                except Exception as e:
                    logger.warning(f"Social MCP failed: {e}")
    
            # Only embed if we actually got documents
            if docs:
                vectors = [self.embed.encode(d) for d in docs]
                self.store.add(vectors, docs)
    
        # Perform semantic retrieval regardless of MCP
        vec = self.embed.encode(state["query"])
        retrieved_docs = self.store.search(vec)
    
        return {**state, "docs": retrieved_docs}
    
