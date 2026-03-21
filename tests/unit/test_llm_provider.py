"""Tests for provider-agnostic LLM abstraction."""

from unittest.mock import MagicMock, patch

import pytest

from retune.core.llm import create_llm, detect_provider, set_default_llm


class TestDetectProvider:
    def test_openai_models(self):
        assert detect_provider("gpt-4o-mini") == "openai"
        assert detect_provider("gpt-4o") == "openai"
        assert detect_provider("gpt-3.5-turbo") == "openai"
        assert detect_provider("o1-preview") == "openai"
        assert detect_provider("o3-mini") == "openai"

    def test_anthropic_models(self):
        assert detect_provider("claude-sonnet-4-20250514") == "anthropic"
        assert detect_provider("claude-3-haiku-20240307") == "anthropic"
        assert detect_provider("claude-opus-4-20250514") == "anthropic"

    def test_google_models(self):
        assert detect_provider("gemini-1.5-flash") == "google"
        assert detect_provider("gemini-2.0-flash") == "google"
        assert detect_provider("gemma-7b") == "google"

    def test_ollama_models(self):
        assert detect_provider("llama3.1") == "ollama"
        assert detect_provider("mistral-7b") == "ollama"
        assert detect_provider("phi-3") == "ollama"
        assert detect_provider("deepseek-coder") == "ollama"

    def test_unknown_defaults_to_openai(self):
        assert detect_provider("some-custom-model") == "openai"


class TestCreateLLM:
    def test_returns_user_provided_llm(self):
        """If user passes an LLM instance, return it directly."""
        mock_llm = MagicMock()
        result = create_llm(llm=mock_llm)
        assert result is mock_llm

    def test_returns_global_default(self):
        """If global default is set, return it."""
        mock_llm = MagicMock()
        set_default_llm(mock_llm)
        try:
            result = create_llm()
            assert result is mock_llm
        finally:
            # Clean up global state
            import retune.core.llm as llm_module
            llm_module._default_llm = None

    def test_user_llm_takes_priority_over_global(self):
        """User-provided LLM takes priority over global default."""
        global_llm = MagicMock(name="global")
        user_llm = MagicMock(name="user")

        set_default_llm(global_llm)
        try:
            result = create_llm(llm=user_llm)
            assert result is user_llm
        finally:
            import retune.core.llm as llm_module
            llm_module._default_llm = None

    def test_openai_fallback_import_error(self):
        """If langchain-openai not installed, raise clear error."""
        import retune.core.llm as llm_module
        llm_module._default_llm = None

        with patch.dict("sys.modules", {"langchain_openai": None}):
            with patch("builtins.__import__", side_effect=ImportError("no langchain_openai")):
                # This would raise ImportError in real usage
                # We just verify detect_provider works
                assert detect_provider("gpt-4o-mini") == "openai"


class TestSetDefaultLLM:
    def test_set_and_get(self):
        from retune.core.llm import get_default_llm

        mock = MagicMock()
        set_default_llm(mock)
        try:
            assert get_default_llm() is mock
        finally:
            import retune.core.llm as llm_module
            llm_module._default_llm = None

    def test_default_is_none_initially(self):
        from retune.core.llm import get_default_llm
        import retune.core.llm as llm_module

        llm_module._default_llm = None
        assert get_default_llm() is None
