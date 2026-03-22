"""SDK configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class RetuneSettings(BaseSettings):
    """Global SDK configuration via environment variables or .env file."""

    model_config = {"env_prefix": "RETUNE_", "env_file": ".env", "extra": "ignore"}

    storage_path: str = Field(default="retune.db", description="SQLite database path")
    default_mode: str = Field(default="observe", description="Default wrapper mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Evaluation defaults
    eval_llm_model: str = Field(
        default="gpt-4o-mini", description="Default LLM model for LLM judge evaluator"
    )

    # Scoring weights
    weight_correctness: float = Field(default=0.4)
    weight_retrieval: float = Field(default=0.2)
    weight_tool: float = Field(default=0.15)
    weight_latency: float = Field(default=0.1)
    weight_cost: float = Field(default=0.15)

    # Beam Search APO defaults
    beam_width: int = Field(default=2, description="Beam search width (candidates kept per round)")
    beam_branch_factor: int = Field(default=2, description="Rewrites per candidate per round")
    beam_rounds: int = Field(default=2, description="Number of beam search rounds")
    beam_max_rollout_queries: int = Field(default=5, description="Max queries per candidate")
    beam_cost_budget_usd: float = Field(default=0.50, description="Beam search budget (USD)")

    @property
    def storage_full_path(self) -> Path:
        return Path(self.storage_path).resolve()

    def get_weights(self) -> dict[str, float]:
        return {
            "correctness": self.weight_correctness,
            "retrieval": self.weight_retrieval,
            "tool": self.weight_tool,
            "latency": self.weight_latency,
            "cost": self.weight_cost,
        }


settings = RetuneSettings()
