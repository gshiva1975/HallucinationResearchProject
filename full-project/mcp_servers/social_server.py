# mcp_servers/social_server.py

from mcp_servers.base_mcp import BaseMCP

mcp = BaseMCP()


def fetch_social_sentiment(ticker: str) -> list:
    """
    Returns social sentiment signal for a given stock ticker.
    Extend this function to call Reddit (PRAW), Twitter/X API, etc.
    """
    # Placeholder — replace with real social data source
    return [f"{ticker} trending positively on investor forums with sentiment score 0.78"]


mcp.register("fetch_social_sentiment", fetch_social_sentiment)

app = mcp.app
