"""Token cost estimation for LLM calls."""

from __future__ import annotations

# Approximate pricing per 1M tokens (input, output) in USD
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-2.0-flash": (0.10, 0.40),
}


def estimate_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Estimate USD cost for a single LLM call."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        # Try prefix matching
        for key, val in MODEL_PRICING.items():
            if model.startswith(key.split("-")[0]):
                pricing = val
                break
    if pricing is None:
        return 0.0
    input_price, output_price = pricing
    return (
        prompt_tokens * input_price + completion_tokens * output_price
    ) / 1_000_000


def estimate_tokens_from_text(text: str) -> int:
    """Rough token estimate from text length (4 chars per token average)."""
    return max(1, len(text) // 4)
