# Financial Market Agent with Semantic Tool Selection

A LangChain agent that answers stock and market questions using **OpenAI** as the LLM and
**Google Finance** data fetched through **SerpAPI**. Instead of giving the LLM all tools on
every turn, it uses **semantic tool selection** to pick only the most relevant ones for each query.

## How semantic tool selection works

```
user query ──embed──► similarity search over tool index ──► top-k tools ──► llm.bind_tools(top-k)
                              ▲
     tool name + description  │  + example user queries per tool (OpenAI embeddings,
                              │    InMemoryVectorStore)
```

1. Each tool's description **and** several example user questions (`TOOL_EXAMPLES` in
   `finance_agent/tools.py`) are embedded as separate documents tagged with the tool name.
2. The user query is embedded and compared against all documents (cosine similarity).
3. Each tool's score is its best-matching document; the top-k tools above
   `TOOL_SELECTION_MIN_SCORE` are kept (at least one is always returned).
4. Only those tools are bound to the OpenAI model for that turn, which keeps the prompt
   smaller and reduces wrong-tool calls as the toolset grows.

## Tools

| Tool | Source | Purpose |
|---|---|---|
| `get_stock_quote` | Google Finance | Current price & daily change |
| `get_stock_price_history` | Google Finance | Price chart for 1D…MAX, high/low/% change |
| `get_company_key_stats` | Google Finance | Market cap, P/E, dividend yield, 52w range, company info |
| `get_company_financials` | Google Finance | Income statement, balance sheet, cash flow |
| `get_stock_news` | Google Finance | Latest headlines for a ticker |
| `compare_stocks` | Google Finance | Side-by-side quotes for several tickers |
| `get_market_indexes` | Google Finance Markets | S&P 500, Dow, Nasdaq, FTSE, Nikkei, … by region |
| `get_market_movers` | Google Finance Markets | Top gainers / losers / most active |
| `get_crypto_price` | Google Finance | BTC, ETH, SOL, … |
| `get_currency_exchange_rate` | Google Finance | Forex rates |
| `calculate_investment_return` | local | P/L, total return, CAGR |
| `calculate_compound_growth` | local | Future value with monthly contributions |

Tickers use Google Finance format `TICKER:EXCHANGE` (e.g. `AAPL:NASDAQ`, `JPM:NYSE`, `TCS:NSE`).
If no exchange is given, NASDAQ then NYSE are tried.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env          # then put your OPENAI_API_KEY and SERPAPI_API_KEY in .env
```

## Usage

### Chatbot UI (Gradio)

```bash
python app.py
```

Open http://127.0.0.1:7860. The chat shows each tool call as a collapsible 🛠️ step (arguments +
raw data), and the side panel shows which tools semantic selection picked and their similarity
scores. Each browser session gets its own conversation memory, and all sessions share one
tool index, so the tool embeddings are created once per server start.
**Clear conversation** resets the memory.

### Command line

```bash
# One-shot question
python main.py "Compare Apple and Microsoft today and show any news on Apple"

# See which tools the selector would choose (needs only OPENAI_API_KEY, no SerpAPI credits)
python main.py --select-only "What is the USD to INR rate?"

# Interactive chat (keeps conversation history; 'reset' clears it)
python main.py

# Options
python main.py --top-k 3 -v "How has NVDA performed this year?"
```

Output looks like this (numbers illustrative):

```
[semantic tool selection]
  0.712  get_currency_exchange_rate
  0.431  get_crypto_price
  ...
[tool calls]
  -> get_currency_exchange_rate({'from_currency': 'USD', 'to_currency': 'INR'})

1 USD = 84.12 INR (as of ...)
```

### Using it from code

```python
from finance_agent import build_agent

agent = build_agent(top_k=4)
result = agent.invoke("What's Tesla's P/E ratio and market cap?")
print([m.tool.name for m in result.selected_tools])
print(result.answer)
```

## Configuration (`.env`)

| Variable | Default | |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings for tool selection |
| `TOOL_SELECTION_TOP_K` | `4` | Max tools exposed per query |
| `TOOL_SELECTION_MIN_SCORE` | `0.2` | Min cosine similarity for a tool |
| `AGENT_MAX_ITERATIONS` | `6` | Max LLM↔tool round-trips per query |
| `SERPAPI_CACHE_TTL` | `60` | Seconds to cache identical SerpAPI calls |

## Tests

Offline (SerpAPI mocked, fake embeddings and LLM — no keys needed):

```bash
pytest
```

## Project layout

```
finance_agent/
  config.py          settings from .env
  serpapi_client.py  SerpAPI google_finance / google_finance_markets + caching
  tools.py           12 LangChain tools + example queries for the selector
  tool_selector.py   SemanticToolSelector (embeddings + vector store)
  agent.py           tool-calling loop that binds only the selected tools
app.py               Gradio chatbot UI
main.py              CLI
tests/
```

## Adding a tool

Write a `@tool` function in `tools.py` with a specific docstring, add it to `ALL_TOOLS`, and add
2–3 realistic user questions to `TOOL_EXAMPLES`. The selector indexes it automatically.
