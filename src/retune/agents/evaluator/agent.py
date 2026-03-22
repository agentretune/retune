"""Evaluator Deep Agent — Deep Agents v2 with LangChain Deep Agents architecture.

Two execution paths:
1. **Deep Agent mode** (deepagents installed): Uses `create_deep_agent` with
   TodoListMiddleware, SubAgentMiddleware, and FilesystemMiddleware for
   planning, delegation, and large data handling.
2. **Fallback mode** (deepagents not installed): Uses the LangGraph StateGraph
   supervisor pattern with function-based subagents.

Both paths implement BaseEvaluator for drop-in compatibility.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from retune.core.models import EvalResult, ExecutionTrace
from retune.evaluators.base import BaseEvaluator
from retune.tools.builtin.credit_assigner import CreditAssignerTool
from retune.tools.builtin.trace_reader import TraceReaderTool

logger = logging.getLogger(__name__)

# Check if deepagents is available
_HAS_DEEPAGENTS = False
try:
    from deepagents import create_deep_agent
    from deepagents.middleware import (
        FilesystemMiddleware,
        SubAgentMiddleware,
        TodoListMiddleware,
    )
    _HAS_DEEPAGENTS = True
except ImportError:
    pass

# Subagent names
TRACE_ANALYZER = "trace_analyzer"
CREDIT_ASSIGNER = "credit_assigner"
TOOL_AUDITOR = "tool_auditor"
HALLUCINATION_DETECTOR = "hallucination_detector"

ALL_SUBAGENTS = [TRACE_ANALYZER, CREDIT_ASSIGNER, TOOL_AUDITOR, HALLUCINATION_DETECTOR]


def _get_llm(model: str) -> Any:
    """Create LLM instance (provider-agnostic)."""
    from retune.core.llm import create_llm
    return create_llm(model=model, temperature=0)


# ---- Fallback node functions (used when deepagents is not installed) ----

def supervisor_node(state: dict) -> dict:
    """Examine the trace and decide which subagents to run."""
    trace = state["trace"]
    steps = trace.get("steps", [])
    step_types = {s.get("step_type") for s in steps}

    subagents = [TRACE_ANALYZER, CREDIT_ASSIGNER]

    if "tool_call" in step_types:
        subagents.append(TOOL_AUDITOR)
    if "llm_call" in step_types or "retrieval" in step_types:
        subagents.append(HALLUCINATION_DETECTOR)

    return {
        "subagents_to_run": subagents,
        "subagents_completed": [],
    }


def trace_analyzer_node(state: dict) -> dict:
    """Analyze the execution trace using the TraceReaderTool."""
    trace = state["trace"]
    tool = TraceReaderTool()
    analysis = tool.execute(trace=trace)

    completed = list(state.get("subagents_completed", []))
    completed.append(TRACE_ANALYZER)

    return {
        "trace_analysis": analysis,
        "subagents_completed": completed,
    }


def credit_assigner_node(state: dict) -> dict:
    """Run Agent Lightning credit assignment on the trace."""
    trace = state["trace"]
    tool = CreditAssignerTool()

    result = tool.execute(
        steps=trace.get("steps", []),
        response=str(trace.get("response", "")),
        eval_results=trace.get("eval_results", []),
    )

    completed = list(state.get("subagents_completed", []))
    completed.append(CREDIT_ASSIGNER)

    return {
        "credit_assignment": result,
        "subagents_completed": completed,
    }


def tool_auditor_node(state: dict) -> dict:
    """Audit tool usage — were tools called correctly?"""
    trace = state["trace"]
    steps = trace.get("steps", [])
    tool_steps = [s for s in steps if s.get("step_type") == "tool_call"]
    response = str(trace.get("response", ""))

    audit = {
        "total_tool_calls": len(tool_steps),
        "tools_used": [s.get("name", "unknown") for s in tool_steps],
        "tool_details": [],
    }

    for step in tool_steps:
        tool_output = str(step.get("output_data", {}).get("output", ""))
        output_words = [w for w in tool_output.lower().split() if len(w) > 4][:10]
        words_in_response = sum(1 for w in output_words if w in response.lower())
        output_used = words_in_response > 0

        audit["tool_details"].append({
            "name": step.get("name"),
            "output_used_in_response": output_used,
            "output_preview": tool_output[:200],
            "words_matched": words_in_response,
        })

    if not tool_steps:
        audit["score"] = 1.0
        audit["reasoning"] = "No tool calls to audit."
    else:
        used_count = sum(1 for d in audit["tool_details"] if d["output_used_in_response"])
        efficiency = used_count / len(tool_steps) if tool_steps else 1.0
        call_penalty = max(0, (len(tool_steps) - 2) * 0.1)
        audit["score"] = round(max(0, min(1, efficiency - call_penalty)), 2)
        audit["reasoning"] = (
            f"{used_count}/{len(tool_steps)} tool outputs used in response. "
            f"{'Efficient' if len(tool_steps) <= 2 else 'Too many tool calls'}."
        )

    completed = list(state.get("subagents_completed", []))
    completed.append(TOOL_AUDITOR)

    return {
        "tool_audit": audit,
        "subagents_completed": completed,
    }


def hallucination_detector_node(state: dict) -> dict:
    """Detect hallucinations by checking response against source material."""
    trace = state["trace"]
    model = state.get("model", "gpt-4o-mini")

    response = str(trace.get("response", ""))
    steps = trace.get("steps", [])

    sources = []
    for step in steps:
        if step.get("step_type") == "retrieval":
            docs = step.get("output_data", {}).get("documents", [])
            for doc in docs:
                sources.append(str(doc.get("content", "")))
        elif step.get("step_type") == "tool_call":
            output = step.get("output_data", {}).get("output", "")
            if output:
                sources.append(str(output))

    source_text = " ".join(sources).lower()

    check = {"grounded_claims": 0, "ungrounded_claims": 0, "details": []}

    try:
        llm = _get_llm(model)
        prompt = (
            "You are a hallucination detector. Compare the RESPONSE "
            "against the SOURCE material.\n\n"
            f"SOURCE MATERIAL:\n{source_text[:3000]}\n\n"
            f"RESPONSE:\n{response[:2000]}\n\n"
            "List each factual claim in the response. For each claim, state whether it is "
            "'grounded' (supported by source material) or 'ungrounded' (not in sources).\n"
            "Respond in JSON: {\"claims\": [{\"claim\": \"...\", "
            "\"status\": \"grounded|ungrounded\"}], "
            "\"hallucination_score\": 0.0-1.0 where "
            "0=no hallucinations, 1=fully hallucinated}"
        )
        result = llm.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)

        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            check["hallucination_score"] = parsed.get("hallucination_score", 0.5)
            claims = parsed.get("claims", [])
            check["grounded_claims"] = sum(1 for c in claims if c.get("status") == "grounded")
            check["ungrounded_claims"] = sum(1 for c in claims if c.get("status") == "ungrounded")
            check["details"] = claims[:10]
        else:
            check["hallucination_score"] = 0.5
            check["raw_response"] = content[:500]

    except Exception as e:
        logger.warning(f"Hallucination detection failed: {e}")
        hallucination_markers = [
            "blockchain", "quantum", "cryptocurrency", "300%",
            "fastest growing", "recent studies show",
        ]
        found = [m for m in hallucination_markers if m in response.lower()]
        check["hallucination_score"] = min(len(found) * 0.3, 1.0)
        check["ungrounded_claims"] = len(found)
        check["fallback"] = True

    score = round(1.0 - check.get("hallucination_score", 0.5), 2)
    check["score"] = score
    check["reasoning"] = (
        f"{check['grounded_claims']} grounded, {check['ungrounded_claims']} ungrounded claims. "
        f"Grounding score: {score:.2f}"
    )

    completed = list(state.get("subagents_completed", []))
    completed.append(HALLUCINATION_DETECTOR)

    return {
        "hallucination_check": check,
        "subagents_completed": completed,
    }


def synthesizer_node(state: dict) -> dict:
    """Aggregate all subagent analyses into a final EvalResult."""
    trace = state["trace"]
    model = state.get("model", "gpt-4o-mini")

    trace_analysis = state.get("trace_analysis", {})
    credit_assignment = state.get("credit_assignment", {})
    tool_audit = state.get("tool_audit", {})
    hallucination_check = state.get("hallucination_check", {})

    scores = {}
    reasoning_parts = []

    if trace_analysis:
        has_reasoning = trace_analysis.get("has_reasoning", False)
        scores["reasoning_presence"] = 1.0 if has_reasoning else 0.3
        reasoning_parts.append(
            f"Steps: {trace_analysis.get('total_steps', 0)}, "
            f"Duration: {trace_analysis.get('total_duration_ms', 0):.0f}ms, "
            f"Tokens: {trace_analysis.get('total_tokens', 0)}"
        )

    if credit_assignment:
        is_failure = credit_assignment.get("is_failure", False)
        if is_failure:
            bottlenecks = credit_assignment.get("bottlenecks", [])
            if bottlenecks:
                top = bottlenecks[0]
                reasoning_parts.append(
                    f"Primary bottleneck: [{top['step_type']}] {top['name']} "
                    f"(blame={top['blame_score']:.2f})"
                )
        scores["credit_health"] = credit_assignment.get("overall_score", 0.5)

    if tool_audit:
        scores["tool_usage"] = tool_audit.get("score", 1.0)
        reasoning_parts.append(f"Tool audit: {tool_audit.get('reasoning', 'N/A')}")

    if hallucination_check:
        scores["grounding"] = hallucination_check.get("score", 0.5)
        reasoning_parts.append(f"Grounding: {hallucination_check.get('reasoning', 'N/A')}")

    try:
        llm = _get_llm(model)

        analysis_text = json.dumps({
            "query": trace.get("query", ""),
            "response_preview": str(trace.get("response", ""))[:500],
            "trace_analysis": trace_analysis,
            "credit_assignment_summary": credit_assignment.get("summary", ""),
            "tool_audit": tool_audit,
            "hallucination_check": {
                "score": hallucination_check.get("score"),
                "reasoning": hallucination_check.get("reasoning"),
            } if hallucination_check else None,
            "dimension_scores": scores,
        }, default=str, indent=2)

        prompt = (
            "You are an expert AI system evaluator. Based on the following analysis, "
            "provide a final evaluation score and reasoning.\n\n"
            f"ANALYSIS:\n{analysis_text[:4000]}\n\n"
            "Respond in JSON:\n"
            '{"overall_score": <float 0.0-1.0>, "correctness": <float>, '
            '"completeness": <float>, "relevance": <float>, '
            '"tool_efficiency": <float>, "reasoning": "<brief explanation>"}'
        )

        result = llm.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)

        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            final_score = parsed.get("overall_score", 0.5)
            details = {
                "correctness": parsed.get("correctness", final_score),
                "completeness": parsed.get("completeness", final_score),
                "relevance": parsed.get("relevance", final_score),
                "tool_efficiency": parsed.get("tool_efficiency", scores.get("tool_usage", 1.0)),
                "grounding": scores.get("grounding", 1.0),
                "bottlenecks": credit_assignment.get("bottlenecks", []),
            }
            reasoning = parsed.get("reasoning", "; ".join(reasoning_parts))
        else:
            raise ValueError("Could not parse LLM response")

    except Exception as e:
        logger.warning(f"LLM synthesis failed, using heuristic: {e}")
        if scores:
            final_score = round(sum(scores.values()) / len(scores), 2)
        else:
            final_score = 0.5
        details = {**scores}
        reasoning = "; ".join(reasoning_parts) or "Heuristic evaluation"

    return {
        "final_eval": {
            "evaluator_name": "deep_evaluator",
            "score": round(final_score, 2),
            "reasoning": reasoning,
            "details": details,
        }
    }


def _route_next(state: dict) -> str:
    """Route to the next unfinished subagent, or to synthesizer."""
    to_run = state.get("subagents_to_run", [])
    completed = state.get("subagents_completed", [])

    for subagent in to_run:
        if subagent not in completed:
            return subagent

    return "synthesizer"


# ---- Deep Agent builder (used when deepagents IS installed) ----

def _build_deep_evaluator_agent(model: str) -> Any:
    """Build an evaluator using LangChain Deep Agents (create_deep_agent).

    Features:
    - TodoListMiddleware: plans evaluation before acting
    - SubAgentMiddleware: delegates to specialized subagents
    - FilesystemMiddleware: offloads large trace data to files
    """
    from retune.agents.evaluator.prompts import EVALUATOR_MAIN_PROMPT
    from retune.agents.evaluator.subagents.definitions import (
        get_all_evaluator_subagent_definitions,
    )
    from retune.tools.builtin.metrics import MetricsCalculatorTool
    from retune.tools.builtin.trace_reader import TraceReaderTool

    tools = [TraceReaderTool(), MetricsCalculatorTool()]
    langchain_tools = [t.to_langchain_tool() for t in tools]

    subagent_defs = get_all_evaluator_subagent_definitions()
    subagent_configs = []
    for defn in subagent_defs:
        sub_tools = [t.to_langchain_tool() for t in defn["tools"]]
        subagent_configs.append({
            "name": defn["name"],
            "description": defn["description"],
            "system_prompt": defn["system_prompt"],
            "tools": sub_tools,
        })

    from retune.core.llm import create_llm
    llm = create_llm(model=model, temperature=0)

    agent = create_deep_agent(
        model=llm,
        system_prompt=EVALUATOR_MAIN_PROMPT,
        tools=langchain_tools,
        subagents=subagent_configs,
        middleware=[
            TodoListMiddleware(),
            FilesystemMiddleware(),
            SubAgentMiddleware(),
        ],
    )

    return agent


class EvaluatorDeepAgent(BaseEvaluator):
    """Deep Agent v2 evaluator — uses LangChain Deep Agents when available,
    falls back to LangGraph StateGraph supervisor pattern otherwise.

    Deep Agent features:
    - Planning: writes todos before evaluation ("analyze steps", "assign credit", etc.)
    - Delegation: spawns isolated subagents for trace analysis, credit assignment, etc.
    - File system: offloads large trace data to files instead of stuffing context
    - Iterative: subagents plan their own work, use tools iteratively

    Fallback features (LangGraph):
    - Supervisor routing to function-based subagents
    - Sequential execution with state sharing
    - Same analysis capabilities, less sophisticated orchestration

    Usage:
        evaluator = EvaluatorDeepAgent(model="gpt-4o-mini")
        result = evaluator.evaluate(trace)  # Returns EvalResult
    """

    name = "deep_evaluator"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model
        self._graph = None
        self._deep_agent = None
        self._use_deep_agents = _HAS_DEEPAGENTS

    def _build_graph(self) -> Any:
        """Build the LangGraph evaluation graph (fallback)."""
        from langgraph.graph import END, StateGraph

        from retune.agents.evaluator.state import EvaluatorState

        sg = StateGraph(EvaluatorState)

        sg.add_node("supervisor", supervisor_node)
        sg.add_node(TRACE_ANALYZER, trace_analyzer_node)
        sg.add_node(CREDIT_ASSIGNER, credit_assigner_node)
        sg.add_node(TOOL_AUDITOR, tool_auditor_node)
        sg.add_node(HALLUCINATION_DETECTOR, hallucination_detector_node)
        sg.add_node("router", lambda state: {})
        sg.add_node("synthesizer", synthesizer_node)

        sg.set_entry_point("supervisor")
        sg.add_edge("supervisor", "router")

        sg.add_conditional_edges("router", _route_next, {
            TRACE_ANALYZER: TRACE_ANALYZER,
            CREDIT_ASSIGNER: CREDIT_ASSIGNER,
            TOOL_AUDITOR: TOOL_AUDITOR,
            HALLUCINATION_DETECTOR: HALLUCINATION_DETECTOR,
            "synthesizer": "synthesizer",
        })

        for sub in ALL_SUBAGENTS:
            sg.add_edge(sub, "router")

        sg.add_edge("synthesizer", END)

        return sg.compile()

    def _evaluate_with_deep_agent(self, trace: ExecutionTrace) -> EvalResult:
        """Evaluate using LangChain Deep Agents (create_deep_agent)."""
        if self._deep_agent is None:
            self._deep_agent = _build_deep_evaluator_agent(self._model)

        trace_data = trace.model_dump(mode="json")
        trace_json = json.dumps(trace_data, default=str)

        # Deep agent processes the trace — it will:
        # 1. Write todos (plan the evaluation)
        # 2. Delegate to subagents (trace-analyzer, credit-assigner, etc.)
        # 3. Each subagent plans, uses tools, writes findings to file
        # 4. Main agent reads findings, synthesizes final score
        input_message = (
            f"Evaluate this AI agent execution trace:\n\n"
            f"Query: {trace.query}\n"
            f"Response preview: {str(trace.response)[:500]}\n"
            f"Steps: {len(trace.steps)}\n"
            f"Duration: {trace.duration_ms:.0f}ms\n"
            f"Total tokens: {trace.total_tokens}\n\n"
            f"Full trace data has been written to the file system. "
            f"Use your tools to analyze it.\n\n"
            f"TRACE JSON:\n{trace_json[:4000]}"
        )

        result = self._deep_agent.invoke({"messages": [("human", input_message)]})

        # Parse the deep agent's response into an EvalResult
        return self._parse_deep_agent_result(result)

    def _parse_deep_agent_result(self, result: Any) -> EvalResult:
        """Parse the deep agent's output into a structured EvalResult."""
        import re

        # Extract the last AI message
        messages = result.get("messages", [])
        if not messages:
            return EvalResult(
                evaluator_name=self.name,
                score=0.5,
                reasoning="Deep agent returned no messages",
            )

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        # Try to parse JSON from the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return EvalResult(
                    evaluator_name=self.name,
                    score=float(parsed.get("overall_score", parsed.get("score", 0.5))),
                    reasoning=parsed.get("reasoning", content[:500]),
                    details={
                        k: v for k, v in parsed.items()
                        if k not in ("overall_score", "score", "reasoning")
                    },
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: try to extract a score from the text
        score_match = re.search(r'(?:score|rating)[:\s]*(\d+\.?\d*)', content.lower())
        score = float(score_match.group(1)) if score_match else 0.5
        if score > 1.0:
            score = score / 10.0  # Handle 0-10 scale
        score = max(0.0, min(1.0, score))

        return EvalResult(
            evaluator_name=self.name,
            score=score,
            reasoning=content[:500],
        )

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        """Evaluate a trace using the deep agent. Implements BaseEvaluator.

        Tries Deep Agent mode first, falls back to LangGraph if unavailable.
        """
        # Try Deep Agent mode
        if self._use_deep_agents:
            try:
                return self._evaluate_with_deep_agent(trace)
            except Exception as e:
                logger.warning(f"Deep Agent evaluation failed, falling back to LangGraph: {e}")
                self._use_deep_agents = False

        # Fallback: LangGraph StateGraph
        if self._graph is None:
            try:
                self._graph = self._build_graph()
            except ImportError as e:
                logger.error(f"LangGraph not available: {e}")
                return self._heuristic_evaluate(trace)

        initial_state = {
            "trace": trace.model_dump(mode="json"),
            "subagents_to_run": [],
            "subagents_completed": [],
            "trace_analysis": {},
            "credit_assignment": {},
            "tool_audit": {},
            "hallucination_check": {},
            "final_eval": {},
            "messages": [],
            "model": self._model,
        }

        try:
            result = self._graph.invoke(initial_state)
            final = result.get("final_eval", {})
            return EvalResult(
                evaluator_name=final.get("evaluator_name", self.name),
                score=final.get("score", 0.5),
                reasoning=final.get("reasoning", "Deep evaluation completed"),
                details=final.get("details", {}),
            )
        except Exception as e:
            logger.error(f"LangGraph evaluator failed: {e}")
            return self._heuristic_evaluate(trace)

    def _heuristic_evaluate(self, trace: ExecutionTrace) -> EvalResult:
        """Pure heuristic fallback when no LLM/framework is available."""
        tool = TraceReaderTool()
        analysis = tool.execute(trace=trace.model_dump(mode="json"))

        credit_tool = CreditAssignerTool()
        credit = credit_tool.execute(
            steps=[s.model_dump(mode="json") for s in trace.steps],
            response=str(trace.response),
            eval_results=[r.model_dump(mode="json") for r in trace.eval_results],
        )

        scores = []
        if analysis.get("has_reasoning"):
            scores.append(0.8)
        else:
            scores.append(0.4)
        scores.append(credit.get("overall_score", 0.5))

        final_score = sum(scores) / len(scores) if scores else 0.5

        return EvalResult(
            evaluator_name=self.name,
            score=round(final_score, 2),
            reasoning=f"Heuristic evaluation: {analysis.get('total_steps', 0)} steps, "
                      f"credit score {credit.get('overall_score', 0.5):.2f}",
            details={
                "trace_analysis": analysis,
                "credit_assignment": credit,
                "mode": "heuristic_fallback",
            },
        )
