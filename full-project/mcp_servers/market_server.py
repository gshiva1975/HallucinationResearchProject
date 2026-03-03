# mcp_servers/market_server.py

from mcp_servers.base_mcp import BaseMCP
import requests

mcp = BaseMCP()

# https://www.alphavantage.co/support/#api-key
#ALPHA_VANTAGE_API_KEY = "N3LC8ZP2KA0DOEFX"

import os

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def fetch_market_data_1(ticker: str):

    url = (
        f"https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    )

    response = requests.get(url)
    if response.status_code != 200:
        return []

    data = response.json()

    if "Time Series (Daily)" not in data:
        return []

    time_series = data["Time Series (Daily)"]

    # Get latest trading day
    latest_date = sorted(time_series.keys())[-1]
    latest_data = time_series[latest_date]

    open_price = latest_data["1. open"]
    high = latest_data["2. high"]
    low = latest_data["3. low"]
    close = latest_data["4. close"]
    volume = latest_data["5. volume"]

    summary = (
        f"{ticker} market summary for {latest_date}: "
        f"Opened at ${open_price}, "
        f"High ${high}, Low ${low}, "
        f"Closed at ${close}, "
        f"Volume {volume}."
    )

    return [summary]



def fetch_market_data(ticker: str):
    if not ALPHA_VANTAGE_API_KEY:
        return ["API key not configured"]

    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY"
        f"&symbol={ticker}"
        f"&apikey={ALPHA_VANTAGE_API_KEY}"
    )

    response = requests.get(url)
    if response.status_code != 200:
        return [f"Failed to fetch market data for {ticker}"]

    data = response.json()

    if "Time Series (Daily)" not in data:
        return ["Market data unavailable (rate limit or API issue)"]

    latest_date = sorted(data["Time Series (Daily)"].keys(), reverse=True)[0]
    latest = data["Time Series (Daily)"][latest_date]

    summary = (
        f"{ticker} ({latest_date}) — "
        f"Open: ${latest['1. open']}, "
        f"High: ${latest['2. high']}, "
        f"Low: ${latest['3. low']}, "
        f"Close: ${latest['4. close']}, "
        f"Volume: {latest['5. volume']}"
    )

    return [summary]
mcp.register("fetch_market_data", fetch_market_data)

app = mcp.app
