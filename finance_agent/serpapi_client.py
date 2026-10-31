"""Thin SerpAPI wrapper for the Google Finance engines.

Docs:
  https://serpapi.com/google-finance-api
  https://serpapi.com/google-finance-markets
"""

import os
import time
from typing import Any

import requests

SERPAPI_URL = "https://serpapi.com/search.json"
_CACHE: dict[tuple, tuple[float, dict]] = {}


class SerpApiError(RuntimeError):
    pass


def search(engine: str, **params: Any) -> dict:
    """Call SerpAPI with a short-lived in-memory cache keyed on the request params."""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise SerpApiError("SERPAPI_API_KEY is not set")

    query = {"engine": engine, **{k: v for k, v in params.items() if v is not None}}
    key = tuple(sorted(query.items()))
    ttl = float(os.getenv("SERPAPI_CACHE_TTL", "60"))
    cached = _CACHE.get(key)
    if cached and time.monotonic() - cached[0] < ttl:
        return cached[1]

    try:
        resp = requests.get(SERPAPI_URL, params={**query, "api_key": api_key}, timeout=30)
    except requests.RequestException as exc:
        raise SerpApiError(f"SerpAPI request failed: {exc}") from exc
    try:
        data = resp.json()
    except ValueError as exc:
        raise SerpApiError(f"SerpAPI returned non-JSON response (HTTP {resp.status_code})") from exc
    if "error" in data:
        raise SerpApiError(data["error"])
    if not resp.ok:
        raise SerpApiError(f"SerpAPI HTTP {resp.status_code}")

    _CACHE[key] = (time.monotonic(), data)
    return data


def google_finance(q: str, **params: Any) -> dict:
    return search("google_finance", q=q, **params)


def google_finance_markets(trend: str, index_market: str | None = None) -> dict:
    return search("google_finance_markets", trend=trend, index_market=index_market)


def resolve_ticker(ticker: str, **params: Any) -> dict:
    """Fetch a Google Finance page for a ticker, guessing the exchange if none was given.

    Google Finance identifies equities as TICKER:EXCHANGE (e.g. AAPL:NASDAQ).
    """
    ticker = ticker.strip().upper()
    if ":" in ticker or "-" in ticker:
        candidates = [ticker]
    else:
        candidates = [f"{ticker}:NASDAQ", f"{ticker}:NYSE", ticker]

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data = google_finance(candidate, **params)
        except SerpApiError as exc:
            last_error = exc
            continue
        if data.get("summary"):
            return data
    raise SerpApiError(
        f"No Google Finance data for '{ticker}'. Use TICKER:EXCHANGE format "
        f"(e.g. AAPL:NASDAQ, RELIANCE:NSE). Last error: {last_error}"
    )
