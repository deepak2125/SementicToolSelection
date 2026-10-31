"""Environment-driven configuration."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    top_k: int = int(os.getenv("TOOL_SELECTION_TOP_K", "4"))
    min_score: float = float(os.getenv("TOOL_SELECTION_MIN_SCORE", "0.2"))
    max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "6"))


def require_env(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise SystemExit(
            f"Missing environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )
