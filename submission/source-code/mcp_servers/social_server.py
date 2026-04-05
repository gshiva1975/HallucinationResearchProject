# mcp_servers/social_server.py
#
# Social sentiment via Twitter/X API v2.
#
# Required environment variable:
#   TWITTER_BEARER_TOKEN
#
# If not configured, returns a no-data string that is rejected
# by validate_node's has_number check — preventing fabricated grounding.

import os
import logging
import requests

from mcp_servers.base_mcp import BaseMCP

logger = logging.getLogger("SocialServer")

mcp = BaseMCP()

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")


def _vader_score(texts: list) -> dict:
    """Return compound VADER score and label for a list of texts."""
    try:
        import nltk
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

        sia    = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(t)["compound"] for t in texts if t.strip()]
        if not scores:
            return {"label": "neutral", "score": 0.0, "n": 0}

        avg   = sum(scores) / len(scores)
        label = "positive" if avg >= 0.05 else ("negative" if avg <= -0.05 else "neutral")
        return {"label": label, "score": round(avg, 4), "n": len(scores)}

    except Exception as e:
        logger.error(f"VADER scoring failed: {e}")
        return {"label": "neutral", "score": 0.0, "n": 0}


def fetch_social_sentiment(ticker: str) -> list:
    """
    Fetch recent tweets mentioning $TICKER and score with VADER.
    Returns a single summary string, or a no-data string if the
    bearer token is missing or the API call fails.
    """
    logger.info(f"fetch_social_sentiment called  ticker={ticker}")

    if not TWITTER_BEARER_TOKEN:
        logger.warning("TWITTER_BEARER_TOKEN not set — no social data available")
        return [f"{ticker}: social sentiment data unavailable — TWITTER_BEARER_TOKEN not configured"]

    url     = "https://api.twitter.com/2/tweets/search/recent"
    params  = {"query": f"${ticker} lang:en -is:retweet", "max_results": 20, "tweet.fields": "text"}
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        if "errors" in data:
            logger.error(f"Twitter API error: {data['errors']}")
            return [f"{ticker}: Twitter API returned an error — {data['errors']}"]

        tweets = [t["text"] for t in data.get("data", [])]
        logger.info(f"Twitter: fetched {len(tweets)} tweets for ${ticker}")

        if not tweets:
            return [f"{ticker}: no recent tweets found"]

        result  = _vader_score(tweets)
        summary = (
            f"{ticker} social sentiment (Twitter/X, n={result['n']} tweets): "
            f"label={result['label']}, "
            f"compound_score={result['score']}, "
            f"sample_tweet=\"{tweets[0][:120]}\""
        )
        logger.info(f"Sentiment result: {summary}")
        return [summary]

    except Exception as e:
        logger.error(f"Twitter request failed: {e}")
        return [f"{ticker}: Twitter request failed — {e}"]


mcp.register("fetch_social_sentiment", fetch_social_sentiment)

app = mcp.app
