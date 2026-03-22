"""Prompt rewriter tool — LLM-powered prompt rewriting for APO."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from retune.tools.base import RetuneTool

logger = logging.getLogger(__name__)


class PromptRewriterTool(RetuneTool):
    """Rewrites a system prompt based on a textual gradient (critique).

    Used in APO Step 3: takes the current prompt + critique and produces
    an improved prompt. The rewrite moves the prompt in the "opposite direction"
    of the identified failures.
    """

    name: str = "prompt_rewriter"
    description: str = (
        "Rewrite a system prompt based on critique feedback (textual gradient). "
        "Input: current_prompt, critique, optional style hints. "
        "Output: rewritten_prompt, changes_made, confidence."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "current_prompt": {
                "type": "string",
                "description": "The current system prompt to improve",
            },
            "critique": {
                "type": "string",
                "description": "Textual gradient — specific critique of what to fix",
            },
            "style": {
                "type": "string",
                "description": (
                    "Rewrite style: 'conservative' (minimal changes) "
                    "or 'aggressive' (major rewrite)"
                ),
                "default": "conservative",
            },
        },
        "required": ["current_prompt", "critique"],
    }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        current_prompt = kwargs.get("current_prompt", "")
        critique = kwargs.get("critique", "")
        style = kwargs.get("style", "conservative")

        if not current_prompt and not critique:
            return {
                "error": "Both current_prompt and critique are empty",
                "rewritten_prompt": "",
                "confidence": 0.0,
            }

        try:
            from retune.core.llm import create_llm

            llm = create_llm(temperature=0.7)

            style_instruction = (
                "Make minimal, targeted changes — preserve the original structure."
                if style == "conservative"
                else "Feel free to restructure and significantly rewrite the prompt."
            )

            prompt = (
                "You are an expert prompt engineer. "
                "Rewrite the system prompt to address the critique.\n\n"
                f"CURRENT PROMPT:\n\"\"\"\n{current_prompt}\n\"\"\"\n\n"
                f"CRITIQUE (textual gradient):\n{critique[:2000]}\n\n"
                f"STYLE: {style_instruction}\n\n"
                "REQUIREMENTS:\n"
                "1. Address every critique point\n"
                "2. Keep the prompt concise but complete (100-300 words)\n"
                "3. Include: role definition, step-by-step instructions, "
                "constraints, output format\n"
                "4. If tools are mentioned, include specific tool usage guidelines\n"
                "5. Add reasoning instructions if appropriate\n\n"
                "Respond in JSON:\n"
                '{"rewritten_prompt": "<the improved prompt>", '
                '"changes_made": ["<list of specific changes>"], '
                '"confidence": <float 0.0-1.0>}'
            )

            result = llm.invoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "rewritten_prompt": parsed.get("rewritten_prompt", ""),
                    "changes_made": parsed.get("changes_made", []),
                    "confidence": parsed.get("confidence", 0.6),
                }

            # Fallback: treat entire response as rewritten prompt
            return {
                "rewritten_prompt": content.strip(),
                "changes_made": ["Full rewrite (could not parse structured response)"],
                "confidence": 0.4,
            }

        except ImportError:
            logger.warning("LLM provider not available, using heuristic rewrite")
            return self._heuristic_rewrite(current_prompt, critique)
        except Exception as e:
            logger.warning(f"LLM rewrite failed: {e}")
            return self._heuristic_rewrite(current_prompt, critique)

    def _heuristic_rewrite(self, prompt: str, critique: str) -> dict[str, Any]:
        """Fallback heuristic rewrite when LLM is unavailable."""
        additions = []

        critique_lower = critique.lower()
        if "role" in critique_lower and "you are" not in prompt.lower():
            additions.append("You are a helpful AI assistant.")
        if "step" in critique_lower and "step" not in prompt.lower():
            additions.append("Think step-by-step before answering.")
        if "format" in critique_lower and "format" not in prompt.lower():
            additions.append("Provide your response in a clear, structured format.")
        if "constraint" in critique_lower or "guardrail" in critique_lower:
            additions.append("Only use information from provided sources. Do not hallucinate.")
        if "tool" in critique_lower and "tool" not in prompt.lower():
            additions.append("Use available tools when needed to gather information.")

        if additions:
            rewritten = prompt.rstrip() + "\n\n" + "\n".join(additions)
        else:
            rewritten = prompt

        return {
            "rewritten_prompt": rewritten,
            "changes_made": additions or ["No heuristic changes applicable"],
            "confidence": 0.3,
        }
