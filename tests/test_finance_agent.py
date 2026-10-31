"""Offline tests: SerpAPI is mocked, embeddings and LLM are fakes. No API keys needed."""

import json
import re

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from finance_agent import ALL_TOOLS, TOOL_EXAMPLES, FinancialMarketAgent, SemanticToolSelector
from finance_agent import serpapi_client, tools

QUOTE = {
    "summary": {
        "title": "Apple Inc", "stock": "AAPL", "exchange": "NASDAQ", "price": "$230.10",
        "extracted_price": 230.1, "currency": "$",
        "price_movement": {"percentage": 1.2, "value": 2.7, "movement": "Up"},
    },
    "graph": [{"price": 200.0, "date": "d1", "volume": 1}, {"price": 250.0, "date": "d2"}, {"price": 230.0, "date": "d3"}],
    "knowledge_graph": {"key_stats": {"stats": [{"label": "Market cap", "value": "3.5T USD"}]}},
    "news_results": [{"title": "Top stories", "items": [{"title": "Apple beats", "source": "X"}]}, {"title": "Solo", "source": "Y"}],
}


@pytest.fixture
def fake_serp(monkeypatch):
    calls = []

    def fake_google_finance(q, **params):
        calls.append(q)
        if q == "AAPL:NASDAQ":
            return QUOTE
        raise serpapi_client.SerpApiError("no results")

    monkeypatch.setattr(serpapi_client, "google_finance", fake_google_finance)
    return calls


def test_resolve_ticker_tries_exchanges(fake_serp):
    out = json.loads(tools.get_stock_quote.invoke({"ticker": "aapl"}))
    assert out["extracted_price"] == 230.1
    assert fake_serp == ["AAPL:NASDAQ"]


def test_unknown_ticker_returns_error_text(fake_serp):
    out = tools.get_stock_quote.invoke({"ticker": "ZZZZ"})
    assert out.startswith("Error:")
    assert fake_serp == ["ZZZZ:NASDAQ", "ZZZZ:NYSE", "ZZZZ"]


def test_price_history_stats(fake_serp):
    out = json.loads(tools.get_stock_price_history.invoke({"ticker": "AAPL:NASDAQ", "window": "1Y"}))
    assert (out["high"], out["low"], out["change_pct"]) == (250.0, 200.0, 15.0)


def test_key_stats_and_news(fake_serp):
    stats = json.loads(tools.get_company_key_stats.invoke({"ticker": "AAPL:NASDAQ"}))
    assert stats["key_stats"] == {"Market cap": "3.5T USD"}
    news = json.loads(tools.get_stock_news.invoke({"ticker": "AAPL:NASDAQ"}))
    assert [n["title"] for n in news] == ["Apple beats", "Solo"]


def test_calculators():
    r = json.loads(tools.calculate_investment_return.invoke(
        {"amount_invested": 1000, "buy_price": 100, "current_price": 121, "years_held": 2}))
    assert r["profit_loss"] == 210 and r["annualized_return_pct"] == 10.0
    g = json.loads(tools.calculate_compound_growth.invoke({"principal": 1000, "annual_rate_pct": 0, "years": 2, "monthly_contribution": 10}))
    assert g["future_value"] == 1240


class KeywordEmbeddings(Embeddings):
    """Bag-of-words embedding over a fixed vocabulary: crude but deterministic 'semantics'."""

    VOCAB = ["price", "bitcoin", "crypto", "exchange", "currency", "news", "revenue",
             "market", "compare", "invest", "dividend", "history", "gainers"]

    def _embed(self, text: str) -> list[float]:
        words = re.findall(r"[a-z]+", text.lower())
        vec = [float(sum(w.startswith(v) for w in words)) for v in self.VOCAB]
        return vec if any(vec) else [1e-6] * len(self.VOCAB)

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def selector():
    return SemanticToolSelector(ALL_TOOLS, KeywordEmbeddings(), examples=TOOL_EXAMPLES, top_k=3)


def test_every_tool_has_examples():
    assert {t.name for t in ALL_TOOLS} == set(TOOL_EXAMPLES)


def test_selector_picks_relevant_tool(selector):
    assert selector.select("bitcoin crypto price")[0].tool.name == "get_crypto_price"
    assert selector.select("currency exchange rate")[0].tool.name == "get_currency_exchange_rate"
    assert len(selector.select("bitcoin")) <= 3


def test_selector_always_include_and_min_score():
    sel = SemanticToolSelector(ALL_TOOLS, KeywordEmbeddings(), examples=TOOL_EXAMPLES, top_k=1,
                               min_score=0.99, always_include=["calculate_compound_growth"])
    names = [m.tool.name for m in sel.select("bitcoin crypto price")]
    assert names[-1] == "calculate_compound_growth"
    assert len(names) == 2  # min_score fallback keeps the single best match


class ScriptedLLM:
    """Minimal stand-in for a chat model: returns queued AIMessages and records bound tools."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bound = []

    def bind_tools(self, tools, **kwargs):
        self.bound.append([t.name for t in tools])
        return self

    def invoke(self, messages):
        return self.responses.pop(0)


def test_agent_runs_tool_loop_with_selected_tools(selector):
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[{"name": "calculate_investment_return", "id": "1",
                                   "args": {"amount_invested": 100, "buy_price": 10, "current_price": 20}}]),
        AIMessage("You doubled your money."),
    ])
    agent = FinancialMarketAgent(llm, selector)
    result = agent.invoke("I invest 100 at price 10, price now 20")
    assert result.answer == "You doubled your money."
    assert "calculate_investment_return" in llm.bound[0]
    assert len(llm.bound[0]) <= 3
    assert json.loads(result.tool_calls[0]["output"])["profit_loss"] == 100
    assert len(agent.history) == 2


def test_gradio_respond_renders_tool_calls_and_selection(selector, monkeypatch):
    import app

    monkeypatch.setattr(app, "_selector", selector)
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[{"name": "calculate_compound_growth", "id": "1",
                                   "args": {"principal": 1000, "annual_rate_pct": 0, "years": 1}}]),
        AIMessage("You'd have $1,000."),
    ])
    *_, (textbox, history, agent, selection) = app.respond("invest 1000", [], FinancialMarketAgent(llm, selector))
    assert textbox == ""
    assert [m["role"] for m in history] == ["user", "assistant", "assistant"]
    assert history[1]["metadata"]["title"].endswith("calculate_compound_growth")
    assert history[2]["content"] == "You'd have $1,000."
    assert "calculate_compound_growth" in selection
    assert app.clear_chat(agent)[0] == [] and agent.history == []


def test_agent_rejects_unselected_tool(selector):
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[{"name": "not_a_tool", "id": "1", "args": {}}]),
        AIMessage("done"),
    ])
    result = FinancialMarketAgent(llm, selector).invoke("bitcoin price")
    assert "not available" in result.tool_calls[0]["output"]
