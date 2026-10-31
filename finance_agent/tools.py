"""Financial market tools backed by Google Finance (via SerpAPI) plus local calculators.

Each tool's docstring becomes its description, which is what the semantic selector
embeds — so descriptions are written to be specific about *when* to use the tool.
"""

import functools
import json
from typing import Literal

from langchain_core.tools import tool

from . import serpapi_client as serp
from .serpapi_client import SerpApiError

Window = Literal["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "MAX"]


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _quote_summary(data: dict) -> dict:
    s = data.get("summary", {})
    keys = ("title", "stock", "exchange", "price", "extracted_price", "currency", "date", "price_movement", "extensions")
    return {k: s[k] for k in keys if k in s}


def _simplify_entries(entries: list[dict], limit: int = 10) -> list[dict]:
    keys = ("stock", "name", "price", "extracted_price", "currency", "price_movement")
    return [{k: e[k] for k in keys if k in e} for e in entries[:limit]]


def _safe(fn):
    """Return API errors to the LLM as text so it can recover (e.g. retry with an exchange suffix)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except SerpApiError as exc:
            return f"Error: {exc}"
    return wrapper


# --------------------------------------------------------------------------- #
# Market data tools (Google Finance via SerpAPI)
# --------------------------------------------------------------------------- #

@tool
@_safe
def get_stock_quote(ticker: str) -> str:
    """Get the current/latest stock price and today's price change for a single company.

    Use for questions like "what is Apple trading at" or "how is TSLA doing today".
    ticker: Google Finance symbol, ideally TICKER:EXCHANGE (e.g. AAPL:NASDAQ, JPM:NYSE, TCS:NSE).
    """
    return _dump(_quote_summary(serp.resolve_ticker(ticker)))


@tool
@_safe
def get_stock_price_history(ticker: str, window: Window = "1M") -> str:
    """Get historical price chart data and performance over a time window for a stock.

    Use for trends, performance over time, highs/lows, or "how has X moved this year".
    window: one of 1D, 5D, 1M, 6M, YTD, 1Y, 5Y, MAX.
    """
    data = serp.resolve_ticker(ticker, window=window)
    graph = data.get("graph", [])
    prices = [p["price"] for p in graph if isinstance(p.get("price"), (int, float))]
    if not prices:
        return _dump({"summary": _quote_summary(data), "note": "No chart data returned."})

    step = max(1, len(graph) // 30)  # keep the payload small for the LLM
    return _dump({
        "ticker": data["summary"].get("stock", ticker),
        "window": window,
        "start_price": prices[0],
        "end_price": prices[-1],
        "high": max(prices),
        "low": min(prices),
        "change_pct": round((prices[-1] - prices[0]) / prices[0] * 100, 2),
        "points": [{"date": p.get("date"), "price": p.get("price"), "volume": p.get("volume")} for p in graph[::step]],
    })


@tool
@_safe
def get_company_key_stats(ticker: str) -> str:
    """Get company fundamentals and key statistics: market cap, P/E ratio, dividend yield,
    52-week range, previous close, employees, CEO, headquarters, and a company description.

    Use for valuation questions or "tell me about company X".
    """
    data = serp.resolve_ticker(ticker)
    kg = data.get("knowledge_graph", {})
    stats = {s.get("label"): s.get("value") for s in kg.get("key_stats", {}).get("stats", [])}
    about = []
    for item in kg.get("about", []):
        entry = {"title": item.get("title")}
        if isinstance(item.get("description"), dict):
            entry["description"] = item["description"].get("snippet")
        if item.get("info"):
            entry["info"] = {i.get("label"): i.get("value") for i in item["info"]}
        about.append(entry)
    return _dump({"quote": _quote_summary(data), "key_stats": stats, "about": about})


@tool
@_safe
def get_company_financials(
    ticker: str,
    statement: Literal["income_statement", "balance_sheet", "cash_flow", "all"] = "all",
) -> str:
    """Get a company's latest financial statements: revenue, net income, EPS, EBITDA,
    profit margin, total assets/liabilities, cash flow, from quarterly/annual reports.

    Use for earnings, profitability, balance sheet, or cash flow questions.
    """
    data = serp.resolve_ticker(ticker)
    wanted = statement.replace("_", " ")
    out = []
    for section in data.get("financials", []):
        title = section.get("title", "")
        if statement != "all" and wanted not in title.lower():
            continue
        results = section.get("results", [])
        if not results:
            continue
        latest = results[0]
        out.append({
            "statement": title,
            "period": latest.get("date"),
            "rows": [{k: r.get(k) for k in ("title", "value", "change")} for r in latest.get("table", [])],
        })
    return _dump(out or {"note": f"No financial statements returned for {ticker}."})


@tool
@_safe
def get_stock_news(ticker: str, limit: int = 5) -> str:
    """Get the latest news headlines and articles about a specific company or stock.

    Use for "why did X drop/rise", recent events, announcements, or sentiment.
    """
    data = serp.resolve_ticker(ticker)
    articles = []
    for item in data.get("news_results", []):
        articles.extend(item.get("items", [item]))
    keys = ("title", "snippet", "source", "date", "link")
    return _dump([{k: a[k] for k in keys if k in a} for a in articles[:limit]])


@tool
@_safe
def compare_stocks(tickers: list[str]) -> str:
    """Compare current prices and daily price changes of several stocks side by side.

    Use when the user names two or more companies, e.g. "compare Microsoft vs Google".
    """
    results = {}
    for t in tickers[:6]:
        try:
            results[t] = _quote_summary(serp.resolve_ticker(t))
        except SerpApiError as exc:
            results[t] = {"error": str(exc)}
    return _dump(results)


@tool
@_safe
def get_market_indexes(
    region: Literal["americas", "europe-middle-east-africa", "asia-pacific"] = "americas",
) -> str:
    """Get major stock market indexes (S&P 500, Dow Jones, Nasdaq, FTSE, DAX, Nikkei, Sensex...)
    for an overall view of how the market is doing today.
    """
    data = serp.google_finance_markets("indexes", index_market=region)
    trends = [
        {"title": t.get("title"), "results": _simplify_entries(t.get("results", []), 15)}
        for t in data.get("market_trends", [])
    ]
    return _dump(trends)


@tool
@_safe
def get_market_movers(
    category: Literal["gainers", "losers", "most-active", "climate-leaders"] = "gainers",
) -> str:
    """Get today's top market movers: biggest stock gainers, biggest losers, or most actively
    traded stocks. Use for "what stocks are up the most today" style questions.
    """
    data = serp.google_finance_markets(category)
    trends = [
        {"title": t.get("title"), "results": _simplify_entries(t.get("results", []), 10)}
        for t in data.get("market_trends", [])
    ]
    return _dump(trends)


@tool
@_safe
def get_crypto_price(symbol: str, currency: str = "USD") -> str:
    """Get the current price of a cryptocurrency such as Bitcoin (BTC), Ethereum (ETH) or Solana (SOL).

    symbol: crypto ticker like BTC, ETH, SOL. currency: quote currency, default USD.
    """
    q = f"{symbol.strip().upper()}-{currency.strip().upper()}"
    return _dump(_quote_summary(serp.google_finance(q)))


@tool
@_safe
def get_currency_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Get the live foreign exchange (forex) rate between two fiat currencies,
    e.g. USD to EUR, GBP to INR. Use for currency conversion questions.
    """
    q = f"{from_currency.strip().upper()}-{to_currency.strip().upper()}"
    return _dump(_quote_summary(serp.google_finance(q)))


# --------------------------------------------------------------------------- #
# Local calculators (no API calls)
# --------------------------------------------------------------------------- #

@tool
def calculate_investment_return(
    amount_invested: float,
    buy_price: float,
    current_price: float,
    years_held: float | None = None,
    dividends_per_share: float = 0.0,
) -> str:
    """Calculate profit/loss, total return % and annualized return (CAGR) of a past investment
    given the amount invested, buy price and current price. Does not fetch prices itself.
    """
    if buy_price <= 0 or amount_invested <= 0:
        return "Error: amount_invested and buy_price must be positive."
    shares = amount_invested / buy_price
    value = shares * (current_price + dividends_per_share)
    total_return = value / amount_invested - 1
    result = {
        "shares": round(shares, 4),
        "current_value": round(value, 2),
        "profit_loss": round(value - amount_invested, 2),
        "total_return_pct": round(total_return * 100, 2),
    }
    if years_held and years_held > 0 and value > 0:
        result["annualized_return_pct"] = round(((value / amount_invested) ** (1 / years_held) - 1) * 100, 2)
    return _dump(result)


@tool
def calculate_compound_growth(
    principal: float,
    annual_rate_pct: float,
    years: int,
    monthly_contribution: float = 0.0,
) -> str:
    """Project the future value of savings or an investment with compound interest and optional
    monthly contributions (SIP / retirement planning). Rate is the expected annual return in percent.
    """
    r = annual_rate_pct / 100 / 12
    n = years * 12
    if r == 0:
        fv = principal + monthly_contribution * n
    else:
        fv = principal * (1 + r) ** n + monthly_contribution * (((1 + r) ** n - 1) / r)
    contributed = principal + monthly_contribution * n
    return _dump({
        "future_value": round(fv, 2),
        "total_contributed": round(contributed, 2),
        "total_growth": round(fv - contributed, 2),
    })


ALL_TOOLS = [
    get_stock_quote,
    get_stock_price_history,
    get_company_key_stats,
    get_company_financials,
    get_stock_news,
    compare_stocks,
    get_market_indexes,
    get_market_movers,
    get_crypto_price,
    get_currency_exchange_rate,
    calculate_investment_return,
    calculate_compound_growth,
]

# Extra example queries per tool. These are embedded alongside each tool's description
# so the selector matches on how users actually phrase questions, not just on docstrings.
TOOL_EXAMPLES: dict[str, list[str]] = {
    "get_stock_quote": [
        "What is the current price of Apple stock?",
        "How much is NVDA trading at right now?",
        "Is Tesla up or down today?",
    ],
    "get_stock_price_history": [
        "How has Amazon stock performed over the last year?",
        "Show me Microsoft's price trend for the past 6 months",
        "What was the 5 day high and low for Meta?",
    ],
    "get_company_key_stats": [
        "What is Google's market cap and P/E ratio?",
        "Does Coca-Cola pay a dividend? What's the yield?",
        "Who is the CEO of Nvidia and where is it headquartered?",
    ],
    "get_company_financials": [
        "What was Apple's revenue and net income last quarter?",
        "Show me Tesla's balance sheet",
        "How profitable is Microsoft? What's its operating cash flow?",
    ],
    "get_stock_news": [
        "Why did Netflix stock drop today?",
        "Any recent news about Boeing?",
        "What are the latest headlines on AMD?",
    ],
    "compare_stocks": [
        "Compare Apple, Microsoft and Google stock prices",
        "Which is doing better today, Coke or Pepsi?",
        "AAPL vs MSFT",
    ],
    "get_market_indexes": [
        "How is the stock market doing today?",
        "What's the S&P 500 and Dow at?",
        "How are Asian markets performing? Nikkei, Sensex",
    ],
    "get_market_movers": [
        "What are the top gaining stocks today?",
        "Which stocks fell the most today?",
        "Most actively traded stocks right now",
    ],
    "get_crypto_price": [
        "What is the price of Bitcoin?",
        "How much is one Ethereum worth in USD?",
        "Current Solana price",
    ],
    "get_currency_exchange_rate": [
        "Convert 100 US dollars to euros",
        "What is the USD to INR exchange rate?",
        "How many yen per British pound?",
    ],
    "calculate_investment_return": [
        "If I invested $1000 in Apple at $150 and it's now $220, how much did I make?",
        "What's my profit if I bought at 50 and the price is 75?",
        "Calculate my annualized return over 3 years",
    ],
    "calculate_compound_growth": [
        "If I invest $500 a month at 8% for 20 years, how much will I have?",
        "Project retirement savings with compound interest",
        "What will $10,000 grow to in 10 years at 7%?",
    ],
}
