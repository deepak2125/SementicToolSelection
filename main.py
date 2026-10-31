"""CLI for the financial market agent.

Examples:
    python main.py "What is Apple trading at and why did it move today?"
    python main.py --select-only "Convert 500 USD to INR"
    python main.py            # interactive chat
"""

import argparse

from finance_agent import AgentResult, build_agent
from finance_agent.config import require_env


def print_selection(matches) -> None:
    print("\n[semantic tool selection]")
    for m in matches:
        print(f"  {m.score:.3f}  {m.tool.name}")


def print_result(result: AgentResult, verbose: bool) -> None:
    print_selection(result.selected_tools)
    if result.tool_calls:
        print("[tool calls]")
        for c in result.tool_calls:
            print(f"  -> {c['name']}({c['args']})")
            if verbose:
                print(f"     {c['output'][:500]}")
    print(f"\n{result.answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial market agent with semantic tool selection")
    parser.add_argument("query", nargs="*", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--top-k", type=int, help="Number of tools to expose to the LLM per query")
    parser.add_argument("--select-only", action="store_true", help="Only show which tools would be selected")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print raw tool outputs")
    args = parser.parse_args()

    # Selection only needs embeddings; answering also needs market data.
    require_env("OPENAI_API_KEY", *([] if args.select_only else ["SERPAPI_API_KEY"]))
    agent = build_agent(top_k=args.top_k)

    def handle(q: str) -> None:
        if args.select_only:
            print_selection(agent.selector.select(q))
        else:
            print_result(agent.invoke(q), args.verbose)

    if args.query:
        handle(" ".join(args.query))
        return

    print("Financial market agent. Type 'exit' to quit, 'reset' to clear history.")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "reset":
            agent.reset()
            continue
        if q:
            handle(q)


if __name__ == "__main__":
    main()
