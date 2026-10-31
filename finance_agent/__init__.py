from .agent import AgentResult, FinancialMarketAgent
from .tool_selector import SemanticToolSelector, ToolMatch
from .tools import ALL_TOOLS, TOOL_EXAMPLES


def build_selector(top_k: int | None = None, min_score: float | None = None) -> SemanticToolSelector:
    """Embed all tools into the selector index (one embeddings API call)."""
    from langchain_openai import OpenAIEmbeddings

    from .config import Settings

    settings = Settings()
    return SemanticToolSelector(
        ALL_TOOLS,
        OpenAIEmbeddings(model=settings.embedding_model),
        examples=TOOL_EXAMPLES,
        top_k=top_k or settings.top_k,
        min_score=settings.min_score if min_score is None else min_score,
    )


def build_agent(
    top_k: int | None = None,
    min_score: float | None = None,
    selector: SemanticToolSelector | None = None,
) -> FinancialMarketAgent:
    """Wire up the OpenAI chat model, selector and agent from settings.

    Pass an existing `selector` to share one tool index across several agents (e.g. UI sessions).
    """
    from langchain_openai import ChatOpenAI

    from .config import Settings

    settings = Settings()
    selector = selector or build_selector(top_k, min_score)
    llm = ChatOpenAI(model=settings.openai_model, temperature=0)
    return FinancialMarketAgent(llm, selector, max_iterations=settings.max_iterations)


__all__ = [
    "ALL_TOOLS",
    "TOOL_EXAMPLES",
    "AgentResult",
    "FinancialMarketAgent",
    "SemanticToolSelector",
    "ToolMatch",
    "build_agent",
    "build_selector",
]
