"""LLM provider abstraction -- provider-agnostic LLM creation.

Retune does NOT force OpenAI. Users can use ANY LangChain-compatible
chat model: OpenAI, Anthropic, Google, Ollama, Azure, Groq, Together, etc.

Three ways to provide an LLM:

1. Pass an LLM instance directly (any BaseChatModel):
     from langchain_anthropic import ChatAnthropic
     llm = ChatAnthropic(model="claude-sonnet-4-20250514")
     wrapper = Retuner(..., llm=llm)

2. Use the default factory with model string (auto-detects provider):
     wrapper = Retuner(..., eval_llm_model="gpt-4o-mini")      # OpenAI
     wrapper = Retuner(..., eval_llm_model="claude-sonnet-4-20250514")  # Anthropic
     wrapper = Retuner(..., eval_llm_model="gemini-1.5-flash") # Google

3. Set the global LLM so all components use it:
     from retune.core.llm import set_default_llm
     set_default_llm(my_custom_llm)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Global default LLM instance -- set by user or auto-created
_default_llm: Any = None

# Provider detection patterns
_PROVIDER_PATTERNS: dict[str, list[str]] = {
    "openai": ["gpt-", "o1-", "o3-", "o4-"],
    "anthropic": ["claude-"],
    "google": ["gemini-", "gemma-"],
    "ollama": ["llama", "mistral", "phi", "qwen", "deepseek", "codellama"],
}


def set_default_llm(llm: Any) -> None:
    """Set the global default LLM used by all Retune components.

    Args:
        llm: Any LangChain-compatible BaseChatModel instance.

    Example:
        from langchain_anthropic import ChatAnthropic
        from retune.core.llm import set_default_llm

        set_default_llm(ChatAnthropic(model="claude-sonnet-4-20250514"))
    """
    global _default_llm
    _default_llm = llm
    logger.info(f"Default LLM set to: {type(llm).__name__}")


def get_default_llm() -> Any | None:
    """Get the global default LLM, if set."""
    return _default_llm


def detect_provider(model: str) -> str:
    """Detect the LLM provider from a model name string.

    Returns: "openai", "anthropic", "google", "ollama", or "openai" (fallback).
    """
    model_lower = model.lower()
    for provider, patterns in _PROVIDER_PATTERNS.items():
        if any(model_lower.startswith(p) for p in patterns):
            return provider
    # Default to OpenAI (most common, and OpenAI-compatible APIs like Together/Groq)
    return "openai"


def create_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    llm: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Create or return an LLM instance. Provider-agnostic.

    Priority:
    1. If `llm` is passed directly, return it as-is (user's exact instance)
    2. If a global default LLM is set, return that
    3. Auto-detect provider from model name and create appropriate instance

    Args:
        model: Model name string (e.g., "gpt-4o-mini", "claude-sonnet-4-20250514")
        temperature: Sampling temperature
        llm: Optional pre-created LLM instance (highest priority)
        **kwargs: Additional kwargs passed to the LLM constructor

    Returns:
        A LangChain-compatible BaseChatModel instance

    Raises:
        ImportError: If the required provider package is not installed
    """
    # Priority 1: User-provided instance
    if llm is not None:
        return llm

    # Priority 2: Global default
    if _default_llm is not None:
        return _default_llm

    # Priority 3: Auto-create from model name
    provider = detect_provider(model)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=temperature, **kwargs)  # type: ignore[call-arg]
        except ImportError:
            raise ImportError(
                f"langchain-anthropic is required for model '{model}'. "
                "Install with: pip install langchain-anthropic"
            )

    elif provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, temperature=temperature, **kwargs)
        except ImportError:
            raise ImportError(
                f"langchain-google-genai is required for model '{model}'. "
                "Install with: pip install langchain-google-genai"
            )

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model, temperature=temperature, **kwargs)
        except ImportError:
            raise ImportError(
                f"langchain-ollama is required for model '{model}'. "
                "Install with: pip install langchain-ollama"
            )

    else:
        # Default: OpenAI (also works for OpenAI-compatible APIs via base_url)
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=temperature, **kwargs)
        except ImportError:
            raise ImportError(
                f"langchain-openai is required for model '{model}'. "
                "Install with: pip install retune[llm]"
            )
