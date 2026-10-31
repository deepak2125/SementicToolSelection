"""Tool-calling agent that only exposes semantically relevant tools to the LLM per turn."""

from dataclasses import dataclass, field
from datetime import date

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .tool_selector import SemanticToolSelector, ToolMatch

SYSTEM_PROMPT = """You are a financial markets assistant with access to live Google Finance data.

Rules:
- Always use tools for prices, stats, news and market data. Never invent numbers.
- Google Finance equity symbols use TICKER:EXCHANGE, e.g. AAPL:NASDAQ, MSFT:NASDAQ, JPM:NYSE,
  KO:NYSE, RELIANCE:NSE, VOD:LON. If a lookup fails, retry with the correct exchange.
- Mention the currency and the data timestamp when available.
- Be concise. Use tables for comparisons.
- If you give an opinion on buying/selling, note it is not financial advice.

Today's date is {today}."""


@dataclass
class AgentResult:
    answer: str
    selected_tools: list[ToolMatch]
    tool_calls: list[dict] = field(default_factory=list)


class FinancialMarketAgent:
    def __init__(self, llm: BaseChatModel, selector: SemanticToolSelector, max_iterations: int = 6):
        self.llm = llm
        self.selector = selector
        self.max_iterations = max_iterations
        # Only user questions and final answers are kept; tool traffic stays within a turn.
        self.history: list[BaseMessage] = []

    def invoke(self, query: str) -> AgentResult:
        selected = self.selector.select(query)
        tools = {m.tool.name: m.tool for m in selected}
        llm_with_tools = self.llm.bind_tools(list(tools.values()))

        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT.format(today=date.today().isoformat())),
            *self.history,
            HumanMessage(query),
        ]
        calls_log: list[dict] = []

        for _ in range(self.max_iterations):
            ai: AIMessage = llm_with_tools.invoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                break
            for call in ai.tool_calls:
                tool = tools.get(call["name"])
                output = tool.invoke(call["args"]) if tool else f"Error: tool '{call['name']}' is not available."
                calls_log.append({"name": call["name"], "args": call["args"], "output": str(output)})
                messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        else:
            # Iteration budget exhausted: force a final answer from what was gathered.
            ai = self.llm.bind_tools(list(tools.values()), tool_choice="none").invoke(messages)

        answer = ai.content if isinstance(ai.content, str) else str(ai.content)
        self.history += [HumanMessage(query), AIMessage(answer)]
        return AgentResult(answer=answer, selected_tools=selected, tool_calls=calls_log)

    def reset(self) -> None:
        self.history.clear()
