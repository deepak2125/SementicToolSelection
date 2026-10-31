"""Gradio chat UI for the financial market agent.

Run:  python app.py   then open http://127.0.0.1:7860
"""

import json
import threading

import gradio as gr

from finance_agent import FinancialMarketAgent, SemanticToolSelector, build_agent, build_selector
from finance_agent.config import require_env

EXAMPLES = [
    "How is the stock market doing today?",
    "What is Apple trading at, and why did it move?",
    "Compare MSFT:NASDAQ, GOOGL:NASDAQ and AMZN:NASDAQ",
    "What's Nvidia's market cap and P/E ratio?",
    "Show Tesla's revenue and net income from the latest quarter",
    "What is the price of Bitcoin and Ethereum?",
    "Convert 1000 USD to INR",
    "If I invest $500 a month at 8% for 20 years, how much will I have?",
]

_selector: SemanticToolSelector | None = None
_selector_lock = threading.Lock()


def shared_selector() -> SemanticToolSelector:
    """Embed the tool index once and share it across all chat sessions."""
    global _selector
    with _selector_lock:
        if _selector is None:
            _selector = build_selector()
    return _selector


def selection_markdown(matches) -> str:
    if not matches:
        return "_Ask a question to see which tools get selected._"
    rows = ["| Score | Tool |", "|---:|---|"]
    rows += [f"| {m.score:.3f} | `{m.tool.name}` |" for m in matches]
    total = len(shared_selector().tools)
    return f"**{len(matches)} of {total} tools** bound to the LLM for this question:\n\n" + "\n".join(rows)


def tool_call_message(call: dict) -> dict:
    """Render a tool call as a collapsible 'thought' bubble in the chat."""
    output = call["output"]
    try:
        output = json.dumps(json.loads(output), indent=2)
    except ValueError:
        pass
    if len(output) > 2500:
        output = output[:2500] + "\n... (truncated)"
    args = json.dumps(call["args"], ensure_ascii=False)
    return {
        "role": "assistant",
        "content": f"**Arguments:** `{args}`\n\n```json\n{output}\n```",
        "metadata": {"title": f"🛠️ {call['name']}", "status": "done"},
    }


def respond(message: str, history: list, agent: FinancialMarketAgent | None):
    message = (message or "").strip()
    if not message:
        yield "", history, agent, gr.skip()
        return

    if agent is None:
        agent = build_agent(selector=shared_selector())

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "", "metadata": {"title": "⏳ Selecting tools and fetching market data...", "status": "pending"}},
    ]
    yield "", history, agent, gr.skip()

    history = history[:-1]
    try:
        result = agent.invoke(message)
    except Exception as exc:  # surface API/auth errors in the chat instead of crashing the UI
        history.append({"role": "assistant", "content": f"⚠️ Error: {exc}"})
        yield "", history, agent, gr.skip()
        return

    history += [tool_call_message(c) for c in result.tool_calls]
    history.append({"role": "assistant", "content": result.answer})
    yield "", history, agent, selection_markdown(result.selected_tools)


def clear_chat(agent: FinancialMarketAgent | None):
    if agent is not None:
        agent.reset()
    return [], agent, selection_markdown(None)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Financial Market Chatbot") as demo:
        gr.Markdown(
            "# 📈 Financial Market Chatbot\n"
            "Live Google Finance data via SerpAPI · OpenAI LLM · semantic tool selection"
        )
        agent_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=560,
                    placeholder="Ask about stocks, indexes, crypto, currencies or investment math.",
                    buttons=["copy"],
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="e.g. What is AAPL trading at today?",
                        show_label=False,
                        scale=5,
                        autofocus=True,
                    )
                    send = gr.Button("Send", variant="primary", scale=1)
                clear = gr.Button("🗑️ Clear conversation", size="sm")
                gr.Examples(EXAMPLES, inputs=msg)

            with gr.Column(scale=2):
                gr.Markdown("### 🔎 Semantic tool selection")
                selection = gr.Markdown(selection_markdown(None))
                with gr.Accordion("How it works", open=False):
                    gr.Markdown(
                        "Each tool's description and example questions are embedded into a vector store. "
                        "Your question is embedded and matched against them, and only the top-k most "
                        "similar tools are given to the LLM for that turn. Tool calls the LLM makes "
                        "appear in the chat as 🛠️ collapsible steps."
                    )

        outputs = [msg, chatbot, agent_state, selection]
        msg.submit(respond, [msg, chatbot, agent_state], outputs)
        send.click(respond, [msg, chatbot, agent_state], outputs)
        clear.click(clear_chat, [agent_state], [chatbot, agent_state, selection])
    return demo


if __name__ == "__main__":
    require_env("OPENAI_API_KEY", "SERPAPI_API_KEY")
    build_ui().queue().launch()
