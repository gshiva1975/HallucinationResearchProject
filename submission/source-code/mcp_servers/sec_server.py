# mcp_servers/sec_server.py

from mcp_servers.base_mcp import BaseMCP
import requests

mcp = BaseMCP()

SEC_HEADERS = {
    "User-Agent": "BananaEnterpriseSystem/1.0 your_email@example.com"
}

CIK_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"

def get_cik_from_ticker(ticker: str):
    try:
        response = requests.get(CIK_LOOKUP_URL, headers=SEC_HEADERS)
        if response.status_code != 200:
            return None

        data = response.json()
        ticker = ticker.upper()

        for entry in data.values():
            if entry["ticker"].upper() == ticker:
                return str(entry["cik_str"]).zfill(10)

        return None
    except Exception:
        return None


def fetch_sec_filings(ticker: str):

    cik = get_cik_from_ticker(ticker)
    if not cik:
        return []

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    response = requests.get(url, headers=SEC_HEADERS)
    if response.status_code != 200:
        return []

    data = response.json()
    filings = []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])

    for form, date in zip(forms, dates):
        if form in ["10-K", "10-Q"]:
            filings.append(
                f"{ticker} filed {form} on {date}. This filing may contain updated financial disclosures."
            )

    return filings[:5]  # Limit to most recent 5


mcp.register("fetch_sec_filings", fetch_sec_filings)

app = mcp.app
