"""Semantic tool selection.

Instead of binding every tool to the LLM on every turn (which bloats the prompt and
increases the chance of the model picking the wrong tool), we:

1. Embed each tool's name + description, plus a few example user queries per tool,
   into a vector store (one document per text, tagged with the tool name).
2. At query time, embed the user's question and run a similarity search.
3. Score each tool by its best-matching document (max cosine similarity).
4. Keep the top-k tools above a minimum score and bind only those to the LLM.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import InMemoryVectorStore


@dataclass
class ToolMatch:
    tool: BaseTool
    score: float


class SemanticToolSelector:
    def __init__(
        self,
        tools: Sequence[BaseTool],
        embeddings: Embeddings,
        examples: dict[str, list[str]] | None = None,
        top_k: int = 4,
        min_score: float = 0.0,
        always_include: Iterable[str] = (),
    ):
        self.tools = {t.name: t for t in tools}
        self.top_k = top_k
        self.min_score = min_score
        self.always_include = [n for n in always_include if n in self.tools]

        docs = []
        for t in tools:
            docs.append(Document(page_content=f"{t.name}: {t.description}", metadata={"tool": t.name}))
            for example in (examples or {}).get(t.name, []):
                docs.append(Document(page_content=example, metadata={"tool": t.name}))
        self._num_docs = len(docs)
        self.store = InMemoryVectorStore(embeddings)
        self.store.add_documents(docs)

    def rank(self, query: str) -> list[ToolMatch]:
        """Score every tool against the query, best first."""
        hits = self.store.similarity_search_with_score(query, k=self._num_docs)
        best: dict[str, float] = {}
        for doc, score in hits:
            name = doc.metadata["tool"]
            best[name] = max(score, best.get(name, float("-inf")))
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return [ToolMatch(self.tools[name], score) for name, score in ranked]

    def select(self, query: str) -> list[ToolMatch]:
        """Return the top-k relevant tools (always at least one) plus any always-included tools."""
        ranked = self.rank(query)
        selected = [m for m in ranked[: self.top_k] if m.score >= self.min_score] or ranked[:1]
        chosen = {m.tool.name for m in selected}
        for name in self.always_include:
            if name not in chosen:
                selected.append(next(m for m in ranked if m.tool.name == name))
        return selected
