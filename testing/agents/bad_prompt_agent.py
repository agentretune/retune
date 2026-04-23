"""Agent with a deliberately vague system prompt + real LLM call.

Use in: 02_trial_prompt.py — PromptOptimizerAgent should find meaningful
improvements to the system prompt.

Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in env.
"""
from __future__ import annotations

import os
from typing import Any


BAD_SYSTEM_PROMPT = "be helpful"

BETTER_SYSTEM_PROMPT_HINT = """\
You are a customer support assistant for a SaaS billing platform. For every
question:
1. Answer concisely (1-3 sentences).
2. Cite relevant policy sections when applicable.
3. If you don't know the answer, say so — do not make up policy details.
4. Use a professional but friendly tone.
"""  # Used only by the verification in 02_trial_prompt.py


class BadPromptAgent:
    """A minimal LLM-backed agent with a vague system prompt.

    Exposes `.system_prompt` that the Retuner wrapper reads via adapter,
    so Retuner._config.system_prompt overrides propagate.
    """

    def __init__(self) -> None:
        self.system_prompt: str = BAD_SYSTEM_PROMPT
        self._llm = self._pick_llm()

    @staticmethod
    def _pick_llm() -> Any:
        """Pick whichever LLM we have keys for."""
        if os.environ.get("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0.3)
        if os.environ.get("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        raise RuntimeError(
            "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY set — cannot build LLM agent."
        )

    def __call__(self, query: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query),
        ]
        resp = self._llm.invoke(messages)
        return str(resp.content)


def make_bad_prompt_agent() -> BadPromptAgent:
    return BadPromptAgent()


SAMPLE_QUERIES = [
    "How do I update my payment method?",
    "What's your refund policy?",
    "Can I cancel my subscription at any time?",
    "Do you offer annual billing discounts?",
    "How do I add a team member to my account?",
    "What happens if my payment fails?",
    "Can I downgrade my plan mid-cycle?",
    "Is there a free trial?",
    "Where do I find my invoices?",
    "How do I change my billing email?",
]
