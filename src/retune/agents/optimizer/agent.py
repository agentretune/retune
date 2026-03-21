"""Optimizer Deep Agent — Deep Agents v2 with Beam Search APO.

Two execution paths:
1. **Deep Agent mode** (deepagents installed): Uses `create_deep_agent` with
   TodoListMiddleware, SubAgentMiddleware, and FilesystemMiddleware for
   planning, delegation, and large data handling. Beam search APO for prompt optimization.
2. **Fallback mode** (deepagents not installed): Uses the LangGraph StateGraph
   with function-based strategy nodes. Single-pass APO.

Both paths implement BaseOptimizer for drop-in compatibility.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from retune.agents.optimizer.beam_config import BeamSearchConfig
from retune.agents.optimizer.beam_search import BeamSearchAPO
from retune.core.models import ExecutionTrace, OptimizationConfig, Suggestion
from retune.optimizers.base import BaseOptimizer
from retune.tools.builtin.config_search import ConfigSearchTool
from retune.tools.builtin.credit_assigner import CreditAssignerTool
from retune.tools.builtin.metrics import MetricsCalculatorTool
from retune.tools.builtin.prompt_analyzer import PromptAnalyzerTool

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

# Strategy names
APO = "apo"
CONFIG_TUNER = "config_tuner"
TOOL_CURATOR = "tool_curator"

ALL_STRATEGIES = [APO, CONFIG_TUNER, TOOL_CURATOR]


def _get_llm(model: str) -> Any:
    """Create LLM instance (provider-agnostic)."""
    from retune.core.llm import create_llm
    return create_llm(model=model, temperature=0)


# ---- Fallback node functions (LangGraph mode) ----

def planner_node(state: dict) -> dict:
    """Analyze traces and decide which optimization strategies to run."""
    traces = state.get("traces", [])
    current_config = state.get("current_config", {})

    if not traces:
        return {"strategies_to_run": [], "strategies_completed": [], "analysis_summary": {}}

    score_lists: dict[str, list[float]] = {}
    for trace in traces:
        for r in trace.get("eval_results", []):
            name = r.get("evaluator_name", "unknown")
            score_lists.setdefault(name, []).append(r.get("score", 0.5))
            for key, val in r.get("details", {}).items():
                if isinstance(val, (int, float)):
                    score_lists.setdefault(key, []).append(float(val))

    avg_scores = {k: sum(v) / len(v) for k, v in score_lists.items() if v}

    strategies = []

    quality_indicators = ["correctness", "answer_quality", "grounding", "deep_evaluator"]
    quality_scores = [avg_scores[k] for k in quality_indicators if k in avg_scores]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 1.0
    if avg_quality < 0.8:
        strategies.append(APO)

    config_indicators = ["retrieval", "latency", "cost", "tool_usage"]
    config_scores = [avg_scores[k] for k in config_indicators if k in avg_scores]
    avg_config = sum(config_scores) / len(config_scores) if config_scores else 1.0
    if avg_config < 0.8 or avg_quality < 0.7:
        strategies.append(CONFIG_TUNER)

    tool_score = avg_scores.get("tool_usage", avg_scores.get("tool_efficiency", 1.0))
    if tool_score < 0.7:
        strategies.append(TOOL_CURATOR)

    if not strategies and avg_quality < 0.9:
        strategies.append(CONFIG_TUNER)

    return {
        "strategies_to_run": strategies,
        "strategies_completed": [],
        "analysis_summary": {
            "avg_scores": avg_scores,
            "avg_quality": round(avg_quality, 3),
            "num_traces": len(traces),
            "strategies_selected": strategies,
        },
    }


def apo_evaluate_node(state: dict) -> dict:
    """APO Step 1: Evaluate the current prompt against failure traces."""
    traces = state.get("traces", [])
    current_config = state.get("current_config", {})

    current_prompt = current_config.get("system_prompt", "")

    analyzer = PromptAnalyzerTool()
    prompt_analysis = analyzer.execute(prompt=current_prompt) if current_prompt else {
        "quality_score": 0.0, "weaknesses": ["No system prompt defined"]
    }

    failures = []
    for trace in traces:
        eval_results = trace.get("eval_results", [])
        scores = [r.get("score", 0.5) for r in eval_results]
        avg = sum(scores) / len(scores) if scores else 0.5
        if avg < 0.7:
            failures.append({
                "query": trace.get("query", ""),
                "response_preview": str(trace.get("response", ""))[:200],
                "scores": {r.get("evaluator_name"): r.get("score") for r in eval_results},
                "avg_score": round(avg, 2),
            })

    evaluation = (
        f"Current prompt analysis:\n{json.dumps(prompt_analysis, indent=2)}\n\n"
        f"Failure examples ({len(failures)} traces with score < 0.7):\n"
        f"{json.dumps(failures[:5], indent=2, default=str)}"
    )

    return {"apo_evaluation": evaluation}


def apo_critique_node(state: dict) -> dict:
    """APO Step 2: Generate 'textual gradient'."""
    model = state.get("model", "gpt-4o-mini")
    evaluation = state.get("apo_evaluation", "")
    current_config = state.get("current_config", {})
    current_prompt = current_config.get("system_prompt", "")

    llm = _get_llm(model)

    prompt = (
        "You are an expert prompt engineer analyzing an AI agent's system prompt.\n\n"
        f"CURRENT SYSTEM PROMPT:\n\"\"\"\n{current_prompt or '(no prompt defined)'}\n\"\"\"\n\n"
        f"EVALUATION DATA:\n{evaluation[:3000]}\n\n"
        "Based on the failures observed, provide a SPECIFIC critique of the system prompt. "
        "Focus on:\n"
        "1. What instructions are MISSING that caused failures?\n"
        "2. What instructions are too VAGUE and need to be more specific?\n"
        "3. What BAD patterns does the current prompt encourage?\n"
        "4. What STRUCTURAL improvements would help (role, constraints, examples, format)?\n\n"
        "Be specific and actionable. Each critique point should suggest a concrete fix."
    )

    try:
        result = llm.invoke(prompt)
        critique = result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        logger.warning(f"APO critique failed: {e}")
        critique = (
            "Could not generate LLM critique. Heuristic assessment: "
            "The prompt may lack specific instructions for tool usage, "
            "reasoning steps, and output formatting."
        )

    return {"apo_critique": critique}


def apo_rewrite_node(state: dict) -> dict:
    """APO Step 3: Rewrite the prompt incorporating the critique."""
    model = state.get("model", "gpt-4o-mini")
    critique = state.get("apo_critique", "")
    current_config = state.get("current_config", {})
    current_prompt = current_config.get("system_prompt", "")

    llm = _get_llm(model)

    prompt = (
        "You are an expert prompt engineer. Rewrite the system prompt based on the critique.\n\n"
        f"CURRENT SYSTEM PROMPT:\n\"\"\"\n{current_prompt or '(no prompt defined)'}\n\"\"\"\n\n"
        f"CRITIQUE:\n{critique[:2000]}\n\n"
        "REQUIREMENTS for the new prompt:\n"
        "1. Address EVERY critique point\n"
        "2. Keep it concise but complete (aim for 100-300 words)\n"
        "3. Include: role definition, step-by-step instructions, constraints, output format\n"
        "4. If the agent uses tools, include specific tool usage guidelines\n"
        "5. If applicable, add a reasoning instruction (think before acting)\n\n"
        "Respond in JSON:\n"
        '{"rewritten_prompt": "<the new system prompt>", '
        '"changes_made": ["<list of specific changes>"], '
        '"confidence": <float 0.0-1.0 how confident you are this is better>}'
    )

    try:
        result = llm.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)

        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            rewritten = parsed.get("rewritten_prompt", "")
            confidence = parsed.get("confidence", 0.7)
        else:
            rewritten = content
            confidence = 0.5

    except Exception as e:
        logger.warning(f"APO rewrite failed: {e}")
        rewritten = ""
        confidence = 0.0

    completed = list(state.get("strategies_completed", []))
    completed.append(APO)

    return {
        "apo_rewritten_prompt": rewritten,
        "apo_confidence": confidence,
        "strategies_completed": completed,
    }


def config_tuner_node(state: dict) -> dict:
    """Tune config parameters based on eval scores and trace analysis."""
    traces = state.get("traces", [])
    current_config = state.get("current_config", {})
    analysis = state.get("analysis_summary", {})
    model = state.get("model", "gpt-4o-mini")

    avg_scores = analysis.get("avg_scores", {})
    suggestions = []

    search_tool = ConfigSearchTool()

    # 1. Retrieval quality -> top_k
    retrieval_score = avg_scores.get("retrieval", avg_scores.get(
        "completeness", avg_scores.get("doc_coverage", 1.0)))
    if retrieval_score < 0.7:
        current_top_k = current_config.get("top_k", 4)
        result = search_tool.execute(
            param_name="top_k",
            current_value=current_top_k,
            direction="increase",
        )
        if result.get("candidates"):
            best = result["candidates"][0]
            suggestions.append({
                "param_name": "top_k",
                "old_value": current_top_k,
                "new_value": best["value"],
                "reasoning": f"Retrieval score is {retrieval_score:.2f}. {best['rationale']}",
                "confidence": best.get("confidence", 0.7),
                "category": "rag",
            })

    # 2. Grounding / Correctness -> temperature
    grounding = avg_scores.get("grounding", avg_scores.get(
        "correctness", avg_scores.get("deep_evaluator", 1.0)))
    current_temp = current_config.get("temperature")
    if grounding < 0.7 and current_temp is not None and current_temp > 0.3:
        result = search_tool.execute(
            param_name="temperature",
            current_value=current_temp,
            direction="decrease",
        )
        if result.get("candidates"):
            best = result["candidates"][0]
            suggestions.append({
                "param_name": "temperature",
                "old_value": current_temp,
                "new_value": best["value"],
                "reasoning": f"Grounding score is {grounding:.2f}. {best['rationale']}",
                "confidence": 0.85,
                "category": "agent",
            })

    # 3. Latency -> reduce max_tokens
    latency_score = avg_scores.get("latency", 1.0)
    if latency_score < 0.5:
        current_max = current_config.get("max_tokens", 2048)
        if current_max > 1024:
            suggestions.append({
                "param_name": "max_tokens",
                "old_value": current_max,
                "new_value": 1024,
                "reasoning": f"Latency score is {latency_score:.2f}. Reducing max tokens to improve speed.",
                "confidence": 0.6,
                "category": "agent",
            })

    # 4. Reasoning strategy
    reasoning_score = avg_scores.get("reasoning", avg_scores.get("reasoning_presence", 1.0))
    current_strategy = current_config.get("reasoning_strategy")
    if reasoning_score < 0.5 and current_strategy != "cot":
        suggestions.append({
            "param_name": "reasoning_strategy",
            "old_value": current_strategy or "none",
            "new_value": "cot",
            "reasoning": f"Reasoning score is {reasoning_score:.2f}. Chain-of-thought improves deliberate decision-making.",
            "confidence": 0.80,
            "category": "agent",
        })

    # 5. Reranker
    if retrieval_score < 0.7 and not current_config.get("use_reranker"):
        suggestions.append({
            "param_name": "use_reranker",
            "old_value": False,
            "new_value": True,
            "reasoning": f"Retrieval score is {retrieval_score:.2f}. Cross-encoder reranking improves precision by 20-40%.",
            "confidence": 0.75,
            "category": "rag",
        })

    # LLM for additional analysis if scores are mixed
    if len(avg_scores) > 2:
        try:
            llm = _get_llm(model)
            analysis_prompt = (
                "You are an AI system optimizer. Given these evaluation scores and config, "
                "suggest parameter changes.\n\n"
                f"Current config: {json.dumps(current_config, default=str)}\n"
                f"Average scores: {json.dumps(avg_scores)}\n"
                f"Already suggested: {[s['param_name'] for s in suggestions]}\n\n"
                "Are there any ADDITIONAL config changes needed that weren't already suggested? "
                "Respond in JSON: {\"additional_suggestions\": [{\"param_name\": \"...\", "
                "\"new_value\": ..., \"reasoning\": \"...\", \"confidence\": 0.0-1.0}]}"
            )
            result = llm.invoke(analysis_prompt)
            content = result.content if hasattr(result, "content") else str(result)

            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                for s in parsed.get("additional_suggestions", []):
                    if s.get("param_name") not in [x["param_name"] for x in suggestions]:
                        s["old_value"] = current_config.get(s.get("param_name"))
                        s["category"] = "agent"
                        suggestions.append(s)
        except Exception as e:
            logger.debug(f"LLM config analysis skipped: {e}")

    completed = list(state.get("strategies_completed", []))
    completed.append(CONFIG_TUNER)

    return {
        "config_suggestions": suggestions,
        "strategies_completed": completed,
    }


def tool_curator_node(state: dict) -> dict:
    """Analyze tool usage patterns and suggest tool changes."""
    traces = state.get("traces", [])
    current_config = state.get("current_config", {})

    tool_stats: dict[str, dict[str, int]] = {}
    for trace in traces:
        for step in trace.get("steps", []):
            if step.get("step_type") == "tool_call":
                name = step.get("name", "unknown")
                if name not in tool_stats:
                    tool_stats[name] = {"calls": 0, "useful": 0, "wasteful": 0}
                tool_stats[name]["calls"] += 1

                output = str(step.get("output_data", {}).get("output", "")).lower()
                response = str(trace.get("response", "")).lower()
                output_words = [w for w in output.split() if len(w) > 4][:10]
                used = any(w in response for w in output_words)
                if used:
                    tool_stats[name]["useful"] += 1
                else:
                    tool_stats[name]["wasteful"] += 1

    suggestions = []
    current_tools = current_config.get("enabled_tools", [])

    for tool_name, stats in tool_stats.items():
        total = stats["calls"]
        wasteful = stats["wasteful"]
        if total > 0 and wasteful / total > 0.7:
            suggestions.append({
                "param_name": "enabled_tools",
                "old_value": f"includes {tool_name}",
                "new_value": f"remove {tool_name}",
                "reasoning": (
                    f"Tool '{tool_name}' was called {total} times but "
                    f"{wasteful} calls ({wasteful/total:.0%}) were wasteful "
                    f"(output not used in response). Consider disabling it."
                ),
                "confidence": 0.7,
                "category": "agent",
            })

    if current_tools and len(current_tools) > 5:
        suggestions.append({
            "param_name": "enabled_tools",
            "old_value": f"{len(current_tools)} tools enabled",
            "new_value": "reduce to 3-4 core tools",
            "reasoning": (
                f"Agent has {len(current_tools)} tools. "
                "Excessive tools cause selection confusion. Limit to core tools."
            ),
            "confidence": 0.65,
            "category": "agent",
        })

    completed = list(state.get("strategies_completed", []))
    completed.append(TOOL_CURATOR)

    return {
        "tool_suggestions": suggestions,
        "strategies_completed": completed,
    }


def aggregator_node(state: dict) -> dict:
    """Aggregate all suggestions from subagents into the final list."""
    all_suggestions = []

    rewritten = state.get("apo_rewritten_prompt", "")
    confidence = state.get("apo_confidence", 0.0)
    current_config = state.get("current_config", {})

    if rewritten and confidence > 0.3:
        all_suggestions.append({
            "param_name": "system_prompt",
            "old_value": (current_config.get("system_prompt", ""))[:100] or "(no prompt)",
            "new_value": rewritten,
            "reasoning": (
                f"APO (Automatic Prompt Optimization): Evaluated current prompt, "
                f"generated critique, and rewrote with {confidence:.0%} confidence. "
                f"Critique: {state.get('apo_critique', '')[:200]}"
            ),
            "confidence": confidence,
            "category": "prompt",
        })

    # Beam search result (if present)
    beam_result = state.get("beam_search_result")
    if beam_result and beam_result.get("improvement", 0) > 0:
        beam_confidence = min(0.95, beam_result.get("best_score", 0.5) + 0.1)
        all_suggestions.append({
            "param_name": "system_prompt",
            "old_value": (current_config.get("system_prompt", ""))[:100] or "(no prompt)",
            "new_value": beam_result["best_prompt"],
            "reasoning": (
                f"Beam Search APO: Explored {beam_result.get('candidates_explored', 0)} candidates "
                f"over {beam_result.get('rounds_completed', 0)} rounds. "
                f"Best score: {beam_result.get('best_score', 0):.3f} "
                f"(+{beam_result.get('improvement', 0):.3f} over baseline). "
                f"{'Verified' if beam_result.get('verified') else 'Unverified'}."
            ),
            "confidence": beam_confidence,
            "category": "prompt",
        })

    for s in state.get("config_suggestions", []):
        all_suggestions.append(s)

    for s in state.get("tool_suggestions", []):
        all_suggestions.append(s)

    # Sort by confidence (highest first)
    all_suggestions.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    # Deduplicate by param_name (keep highest confidence)
    seen = set()
    unique = []
    for s in all_suggestions:
        if s["param_name"] not in seen:
            seen.add(s["param_name"])
            unique.append(s)

    return {"final_suggestions": unique}


def _route_next_strategy(state: dict) -> str:
    """Route to the next unfinished strategy, or to aggregator."""
    to_run = state.get("strategies_to_run", [])
    completed = state.get("strategies_completed", [])

    for strategy in to_run:
        if strategy not in completed:
            if strategy == APO:
                return "apo_evaluate"
            return strategy

    return "aggregator"


# ---- Deep Agent builder (used when deepagents IS installed) ----

def _build_deep_optimizer_agent(model: str) -> Any:
    """Build an optimizer using LangChain Deep Agents (create_deep_agent)."""
    from retune.agents.optimizer.prompts import OPTIMIZER_MAIN_PROMPT
    from retune.agents.optimizer.subagents.definitions import (
        get_all_optimizer_subagent_definitions,
    )
    from retune.tools.builtin.config_search import ConfigSearchTool
    from retune.tools.builtin.metrics import MetricsCalculatorTool
    from retune.tools.builtin.prompt_analyzer import PromptAnalyzerTool
    from retune.tools.builtin.rollout_runner import RolloutRunnerTool

    tools = [ConfigSearchTool(), MetricsCalculatorTool(), PromptAnalyzerTool(), RolloutRunnerTool()]
    langchain_tools = [t.to_langchain_tool() for t in tools]

    subagent_defs = get_all_optimizer_subagent_definitions()
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
        system_prompt=OPTIMIZER_MAIN_PROMPT,
        tools=langchain_tools,
        subagents=subagent_configs,
        middleware=[
            TodoListMiddleware(),
            FilesystemMiddleware(),
            SubAgentMiddleware(),
        ],
    )

    return agent


class OptimizerDeepAgent(BaseOptimizer):
    """Deep Agent v2 optimizer with Beam Search APO.

    Deep Agent features (when deepagents is installed):
    - Planning: writes todos before optimizing
    - Delegation: spawns subagents for critique, rewrite, config tuning, tool curation
    - Beam Search APO: multi-round prompt optimization with verification rollouts
    - File system: offloads large trace data to files

    Fallback features (LangGraph):
    - Planner → strategy routing → aggregation
    - Single-pass APO (evaluate → critique → rewrite)
    - Same config tuning and tool curation capabilities

    Beam Search APO (optional, with adapter + validation_queries):
    - Multiple rounds of critique → rewrite → verify
    - Keeps top K candidates per round
    - Rollout verification against validation queries
    - Returns the best verified prompt

    Usage:
        optimizer = OptimizerDeepAgent(model="gpt-4o-mini")
        suggestions = optimizer.suggest(traces, current_config)

        # With beam search:
        optimizer = OptimizerDeepAgent(model="gpt-4o-mini", beam_config=BeamSearchConfig())
        suggestions = optimizer.suggest(
            traces, current_config,
            adapter=my_adapter,
            validation_queries=["q1", "q2", "q3"],
        )
    """

    name = "deep_optimizer"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        beam_config: BeamSearchConfig | None = None,
    ) -> None:
        self._model = model
        self._beam_config = beam_config
        self._graph = None
        self._deep_agent = None
        self._use_deep_agents = _HAS_DEEPAGENTS

    def _build_graph(self) -> Any:
        """Build the LangGraph optimization graph (fallback)."""
        from langgraph.graph import END, StateGraph

        from retune.agents.optimizer.state import OptimizerState

        sg = StateGraph(OptimizerState)

        sg.add_node("planner", planner_node)
        sg.add_node("apo_evaluate", apo_evaluate_node)
        sg.add_node("apo_critique", apo_critique_node)
        sg.add_node("apo_rewrite", apo_rewrite_node)
        sg.add_node("config_tuner", config_tuner_node)
        sg.add_node("tool_curator", tool_curator_node)
        sg.add_node("router", lambda state: {})
        sg.add_node("aggregator", aggregator_node)

        sg.set_entry_point("planner")
        sg.add_edge("planner", "router")

        sg.add_conditional_edges("router", _route_next_strategy, {
            "apo_evaluate": "apo_evaluate",
            "config_tuner": "config_tuner",
            "tool_curator": "tool_curator",
            "aggregator": "aggregator",
        })

        sg.add_edge("apo_evaluate", "apo_critique")
        sg.add_edge("apo_critique", "apo_rewrite")
        sg.add_edge("apo_rewrite", "router")

        sg.add_edge("config_tuner", "router")
        sg.add_edge("tool_curator", "router")

        sg.add_edge("aggregator", END)

        return sg.compile()

    def _run_beam_search(
        self,
        traces: list[ExecutionTrace],
        current_config: OptimizationConfig,
        adapter: Any | None = None,
        evaluators: list[Any] | None = None,
        validation_queries: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Run Beam Search APO if configured and applicable."""
        if self._beam_config is None:
            return None

        current_prompt = current_config.system_prompt or ""
        if not current_prompt:
            logger.debug("No system prompt — skipping beam search APO")
            return None

        # Gather failure traces
        failure_traces = []
        for trace in traces:
            trace_dict = trace.model_dump(mode="json")
            scores = [r.get("score", 0.5) for r in trace_dict.get("eval_results", [])]
            avg = sum(scores) / len(scores) if scores else 0.5
            if avg < 0.7:
                failure_traces.append(trace_dict)

        if not failure_traces:
            logger.debug("No failure traces — skipping beam search APO")
            return None

        beam_apo = BeamSearchAPO(config=self._beam_config, model=self._model)
        result = beam_apo.search(
            current_prompt=current_prompt,
            failure_traces=failure_traces,
            adapter=adapter,
            evaluators=evaluators,
            validation_queries=validation_queries,
            current_config=current_config,
        )

        return result.model_dump()

    def suggest(
        self,
        traces: list[ExecutionTrace],
        current_config: OptimizationConfig,
        adapter: Any | None = None,
        validation_queries: list[str] | None = None,
    ) -> list[Suggestion]:
        """Generate optimization suggestions. Implements BaseOptimizer.

        Args:
            traces: Recent execution traces with eval results
            current_config: Current optimization config
            adapter: Optional adapter for beam search rollout verification
            validation_queries: Optional queries for rollout testing
        """
        # Try beam search APO first (if configured)
        beam_result = None
        if self._beam_config is not None:
            try:
                beam_result = self._run_beam_search(
                    traces, current_config,
                    adapter=adapter,
                    validation_queries=validation_queries,
                )
            except Exception as e:
                logger.warning(f"Beam search APO failed: {e}")

        # Try Deep Agent mode
        if self._use_deep_agents:
            try:
                return self._suggest_with_deep_agent(
                    traces, current_config, beam_result
                )
            except Exception as e:
                logger.warning(f"Deep Agent optimizer failed, falling back: {e}")
                self._use_deep_agents = False

        # Fallback: LangGraph StateGraph
        if self._graph is None:
            try:
                self._graph = self._build_graph()
            except ImportError:
                logger.info("LangGraph not available, falling back to BasicOptimizer")
                from retune.optimizers.basic import BasicOptimizer
                return BasicOptimizer().suggest(traces, current_config)

        initial_state = {
            "traces": [t.model_dump(mode="json") for t in traces],
            "current_config": current_config.model_dump(mode="json"),
            "strategies_to_run": [],
            "strategies_completed": [],
            "analysis_summary": {},
            "apo_evaluation": "",
            "apo_critique": "",
            "apo_rewritten_prompt": "",
            "apo_confidence": 0.0,
            "config_suggestions": [],
            "tool_suggestions": [],
            "final_suggestions": [],
            "beam_search_result": beam_result,
            "messages": [],
            "model": self._model,
        }

        try:
            result = self._graph.invoke(initial_state)
            suggestions = []
            for s in result.get("final_suggestions", []):
                suggestions.append(Suggestion(
                    param_name=s.get("param_name", "unknown"),
                    old_value=s.get("old_value"),
                    new_value=s.get("new_value"),
                    reasoning=s.get("reasoning", ""),
                    confidence=s.get("confidence", 0.5),
                    category=s.get("category", "general"),
                ))
            return suggestions

        except Exception as e:
            logger.error(f"LangGraph optimizer failed: {e}")
            from retune.optimizers.basic import BasicOptimizer
            logger.info("Falling back to BasicOptimizer")
            return BasicOptimizer().suggest(traces, current_config)

    def _suggest_with_deep_agent(
        self,
        traces: list[ExecutionTrace],
        current_config: OptimizationConfig,
        beam_result: dict[str, Any] | None = None,
    ) -> list[Suggestion]:
        """Generate suggestions using LangChain Deep Agents."""
        if self._deep_agent is None:
            self._deep_agent = _build_deep_optimizer_agent(self._model)

        traces_data = [t.model_dump(mode="json") for t in traces]
        config_data = current_config.model_dump(mode="json")

        # Build input message for the deep agent
        input_parts = [
            f"Optimize this AI system based on {len(traces)} recent traces.\n\n",
            f"Current config: {json.dumps(config_data, default=str)}\n\n",
        ]

        # Summarize traces
        for i, t in enumerate(traces_data[:5]):
            scores = [r.get("score", 0) for r in t.get("eval_results", [])]
            avg = sum(scores) / len(scores) if scores else 0
            input_parts.append(
                f"Trace {i+1}: query='{t.get('query', '')[:50]}', "
                f"avg_score={avg:.2f}, steps={len(t.get('steps', []))}\n"
            )

        if beam_result:
            input_parts.append(
                f"\nBeam Search APO Result: best_score={beam_result.get('best_score', 0):.3f}, "
                f"improvement={beam_result.get('improvement', 0):.3f}, "
                f"candidates={beam_result.get('candidates_explored', 0)}\n"
            )

        input_parts.append(
            "\nAnalyze these traces and generate optimization suggestions. "
            "Use your subagents for critique, rewriting, config tuning, and tool curation."
        )

        input_message = "".join(input_parts)

        result = self._deep_agent.invoke({"messages": [("human", input_message)]})

        return self._parse_deep_agent_suggestions(result, current_config, beam_result)

    def _parse_deep_agent_suggestions(
        self,
        result: Any,
        current_config: OptimizationConfig,
        beam_result: dict[str, Any] | None = None,
    ) -> list[Suggestion]:
        """Parse deep agent output into Suggestion objects."""
        import re

        suggestions = []

        # Add beam search result as a suggestion if present
        if beam_result and beam_result.get("improvement", 0) > 0:
            suggestions.append(Suggestion(
                param_name="system_prompt",
                old_value=(current_config.system_prompt or "")[:100] or "(no prompt)",
                new_value=beam_result["best_prompt"],
                reasoning=(
                    f"Beam Search APO: {beam_result.get('candidates_explored', 0)} candidates, "
                    f"+{beam_result.get('improvement', 0):.3f} improvement"
                ),
                confidence=min(0.95, beam_result.get("best_score", 0.5) + 0.1),
                category="prompt",
            ))

        # Parse agent response
        messages = result.get("messages", [])
        if not messages:
            return suggestions

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        # Try to extract JSON suggestions
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                for s in parsed:
                    if isinstance(s, dict) and "param_name" in s:
                        suggestions.append(Suggestion(
                            param_name=s["param_name"],
                            old_value=s.get("old_value"),
                            new_value=s.get("new_value"),
                            reasoning=s.get("reasoning", ""),
                            confidence=float(s.get("confidence", 0.5)),
                            category=s.get("category", "general"),
                        ))
            except (json.JSONDecodeError, ValueError):
                pass

        return suggestions
