"""
DEMO: Watch Retune diagnose and fix a bad ReAct Agent.

This simulates a LangGraph-style ReAct agent with REAL problems:
- Vague system prompt -> agent doesn't know how to use tools properly
- High temperature (0.9) -> inconsistent, hallucinated tool calls
- Too many tools available -> agent gets confused, picks wrong tools
- No chain-of-thought -> jumps to answers without reasoning

The self-improvement loop:
1. OBSERVE: Collects traces of agent behavior
2. EVALUATE: Scores tool usage, reasoning quality, answer accuracy
3. IMPROVE: Diagnoses problems, suggests fixes (prompt, temp, tools, CoT)
4. RE-EVALUATE: Shows the improvement with the fixed config

No API keys needed - runs entirely locally.

This demonstrates the SAME patterns that work with real LangGraph agents
(see langgraph_react_real.py for the API-connected version).
"""

import time
import random
from datetime import datetime, timezone
from typing import Any

from retune import Retuner, Mode, OptimizationConfig
from retune.core.enums import StepType
from retune.core.models import (
    EvalResult,
    ExecutionTrace,
    Step,
    Suggestion,
    TokenUsage,
)
from retune.evaluators.base import BaseEvaluator
from retune.optimizers.base import BaseOptimizer
from retune.adapters.base import BaseAdapter


# =============================================================================
# 1. SIMULATED TOOLS (like a real ReAct agent would have)
# =============================================================================

TOOLS = {
    "calculator": {
        "description": "Performs math calculations. Input: math expression.",
        "relevant_for": ["calculate", "math", "compute", "how many", "sum", "multiply", "divide", "percentage"],
    },
    "web_search": {
        "description": "Searches the web for current information.",
        "relevant_for": ["what is", "who is", "explain", "define", "current", "latest", "news"],
    },
    "weather_api": {
        "description": "Gets current weather for a city.",
        "relevant_for": ["weather", "temperature", "forecast", "rain", "sunny"],
    },
    "database_lookup": {
        "description": "Looks up internal company data. Input: query.",
        "relevant_for": ["revenue", "employee", "sales", "internal", "company", "department"],
    },
    "code_executor": {
        "description": "Executes Python code snippets.",
        "relevant_for": ["code", "python", "script", "program", "run"],
    },
    "email_sender": {
        "description": "Sends emails. Input: to, subject, body.",
        "relevant_for": ["email", "send", "notify", "message"],
    },
}

# Simulated tool outputs
def run_tool(tool_name: str, tool_input: str, query: str) -> str:
    """Simulate tool execution."""
    if tool_name == "calculator":
        if "percentage" in query.lower() or "%" in query:
            return "Result: 15% of 200 = 30"
        return "Result: 42"
    elif tool_name == "web_search":
        q = query.lower()
        if "machine learning" in q:
            return "Machine learning is a subset of AI focused on algorithms that learn from data to make predictions or decisions."
        elif "react" in q or "agent" in q:
            return "ReAct agents combine reasoning traces with actions, allowing LLMs to solve tasks by thinking step-by-step and using tools."
        elif "capital" in q:
            return "Paris is the capital of France, with a population of about 2.1 million."
        return "Search results: Various information found about the topic."
    elif tool_name == "weather_api":
        return "Weather in Paris: 18C, partly cloudy, humidity 65%."
    elif tool_name == "database_lookup":
        return "Q3 Revenue: $4.2M, up 12% YoY. Top department: Engineering."
    elif tool_name == "code_executor":
        return "Code executed successfully. Output: [1, 2, 3, 4, 5]"
    elif tool_name == "email_sender":
        return "Email sent successfully."
    return "Tool not found."


# =============================================================================
# 2. SIMULATED ReAct AGENT (quality depends on config!)
# =============================================================================

# System prompts - the bad one vs the improved one
PROMPTS = {
    "vague": "You are a helpful assistant. Answer questions using available tools.",
    "structured": (
        "You are a precise AI assistant. Follow these rules:\n"
        "1. THINK first: Analyze what the user is asking and which tool is most relevant.\n"
        "2. ACT: Use exactly ONE tool that best matches the query. Do NOT call irrelevant tools.\n"
        "3. OBSERVE: Read the tool output carefully.\n"
        "4. RESPOND: Synthesize a clear, accurate answer based on the tool output.\n"
        "Never guess or hallucinate. If unsure, say so."
    ),
}


class SimulatedReActAdapter(BaseAdapter):
    """Simulates a LangGraph ReAct agent whose behavior depends on config.

    Bad config -> wrong tool selection, hallucinations, no reasoning.
    Good config -> correct tools, grounded answers, clear reasoning chain.
    """

    def __init__(self, agent: Any, **kwargs: Any) -> None:
        super().__init__(agent=agent, **kwargs)
        self._config = OptimizationConfig()

    def run(self, query: str, config: OptimizationConfig | None = None, **kwargs: Any) -> ExecutionTrace:
        if config:
            self._config = config

        temperature = self._config.temperature if self._config.temperature is not None else 0.9
        system_prompt = self._config.system_prompt or "vague"
        use_cot = self._config.reasoning_strategy == "cot"
        max_tool_calls = self._config.max_tool_calls or 5
        enabled_tools = self._config.enabled_tools or list(TOOLS.keys())

        started_at = datetime.now(timezone.utc)
        steps = []
        query_lower = query.lower()

        # --- Step 0: Reasoning (only if CoT enabled) ---
        if use_cot:
            reasoning_start = datetime.now(timezone.utc)
            time.sleep(0.05)

            # Determine the best tool
            best_tool = self._find_best_tool(query_lower, enabled_tools)
            reasoning_text = (
                f"The user asks: '{query}'. "
                f"Analyzing available tools: {', '.join(enabled_tools)}. "
                f"The best tool for this query is '{best_tool}' because "
                f"it matches the intent of the question. "
                f"I should call it with the relevant input and use its output to respond."
            )

            steps.append(Step(
                step_type=StepType.REASONING,
                name="chain_of_thought",
                input_data={"query": query},
                output_data={"reasoning": reasoning_text},
                started_at=reasoning_start,
                ended_at=datetime.now(timezone.utc),
            ))

        # --- Step 1+: Tool calls ---
        if use_cot:
            # CoT = smart tool selection, usually 1 call
            tools_to_call = [self._find_best_tool(query_lower, enabled_tools)]
        elif system_prompt == "structured" or "structured" in (system_prompt or ""):
            # Structured prompt = decent tool selection
            tools_to_call = [self._find_best_tool(query_lower, enabled_tools)]
        else:
            # Vague prompt + high temp = confused tool selection
            tools_to_call = self._confused_tool_selection(query_lower, temperature, enabled_tools, max_tool_calls)

        tool_outputs = []
        for tool_name in tools_to_call:
            tool_start = datetime.now(timezone.utc)
            time.sleep(0.08)  # Simulate tool latency

            tool_input = query
            output = run_tool(tool_name, tool_input, query)
            tool_outputs.append({"tool": tool_name, "output": output})

            steps.append(Step(
                step_type=StepType.TOOL_CALL,
                name=tool_name,
                input_data={"input": tool_input},
                output_data={"output": output},
                metadata={"tool_description": TOOLS.get(tool_name, {}).get("description", "")},
                started_at=tool_start,
                ended_at=datetime.now(timezone.utc),
            ))

        # --- Final Step: LLM generates answer ---
        llm_start = datetime.now(timezone.utc)
        time.sleep(0.1)

        response = self._generate_response(
            query, tool_outputs, temperature, system_prompt, use_cot
        )

        prompt_tokens = 150 + len(tools_to_call) * 50
        completion_tokens = len(response.split()) * 2
        total_tokens = prompt_tokens + completion_tokens

        steps.append(Step(
            step_type=StepType.LLM_CALL,
            name="gpt-4o-mini",
            input_data={"system_prompt": system_prompt[:100], "tool_outputs": len(tool_outputs)},
            output_data={"response": response},
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            started_at=llm_start,
            ended_at=datetime.now(timezone.utc),
        ))

        ended_at = datetime.now(timezone.utc)
        return ExecutionTrace(
            query=query,
            response=response,
            steps=steps,
            config_snapshot=self._config.to_flat_dict(),
            started_at=started_at,
            ended_at=ended_at,
        )

    def _find_best_tool(self, query_lower: str, enabled_tools: list[str]) -> str:
        """Find the most relevant tool for the query."""
        best_tool = enabled_tools[0]
        best_score = 0
        for tool_name in enabled_tools:
            if tool_name not in TOOLS:
                continue
            keywords = TOOLS[tool_name]["relevant_for"]
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > best_score:
                best_score = score
                best_tool = tool_name
        return best_tool

    def _confused_tool_selection(
        self, query_lower: str, temperature: float, enabled_tools: list[str], max_calls: int
    ) -> list[str]:
        """Bad config -> agent calls wrong/extra tools."""
        best_tool = self._find_best_tool(query_lower, enabled_tools)

        # High temperature = more random extra tool calls
        extra_calls = int(temperature * 3)  # temp 0.9 -> 2 extra calls
        extra_calls = min(extra_calls, max_calls - 1, len(enabled_tools) - 1)

        tools = [best_tool]
        other_tools = [t for t in enabled_tools if t != best_tool]
        if extra_calls > 0 and other_tools:
            tools.extend(random.sample(other_tools, min(extra_calls, len(other_tools))))

        return tools

    def _generate_response(
        self,
        query: str,
        tool_outputs: list[dict],
        temperature: float,
        system_prompt: str,
        use_cot: bool,
    ) -> str:
        """Generate response - quality depends heavily on config."""
        # Find the most relevant tool output
        relevant_output = tool_outputs[0]["output"] if tool_outputs else "No tool output available."

        # Check if we used the RIGHT tool
        query_lower = query.lower()
        best_tool = self._find_best_tool(query_lower, list(TOOLS.keys()))
        used_right_tool = any(t["tool"] == best_tool for t in tool_outputs)

        if use_cot and used_right_tool:
            # Best case: CoT + right tool = accurate, grounded response
            response = f"Based on my analysis, {relevant_output.lower()} "
            response += "This directly answers your question with verified information."

        elif used_right_tool and temperature < 0.4:
            # Good: right tool + low temp = grounded but less detailed
            response = relevant_output

        elif used_right_tool and temperature >= 0.4:
            # OK: right tool but high temp = adds some hallucinated extras
            response = relevant_output
            if temperature > 0.7 and random.random() < 0.5:
                response += " Additionally, according to recent studies, this has increased by 300% in the last year, making it the fastest growing field in technology."

        elif not used_right_tool and temperature > 0.6:
            # Bad: wrong tool + high temp = confused response
            all_outputs = " ".join(t["output"] for t in tool_outputs)
            response = f"Here's what I found: {all_outputs} "
            if random.random() < 0.4:
                response += "Note: I'm also aware that quantum entanglement plays a role in modern computing architectures."

        else:
            # Wrong tool, low temp = at least doesn't hallucinate
            response = f"Based on available information: {relevant_output}"

        return response

    def get_config(self) -> OptimizationConfig:
        return self._config.model_copy()

    def apply_config(self, config: OptimizationConfig) -> None:
        self._config = config


# =============================================================================
# 3. AGENT-SPECIFIC EVALUATORS
# =============================================================================

class ToolUsageEvaluator(BaseEvaluator):
    """Scores whether the agent used the RIGHT tools, and not too many."""
    name = "tool_usage"

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        query_lower = (trace.query or "").lower()
        tool_steps = [s for s in trace.steps if s.step_type == StepType.TOOL_CALL]

        if not tool_steps:
            return EvalResult(
                evaluator_name=self.name, score=0.2,
                reasoning="No tools were called - agent should use tools",
                details={"tool_usage": 0.2},
            )

        # Find what the best tool should have been
        best_tool = None
        best_score = 0
        for tool_name, info in TOOLS.items():
            kw_score = sum(1 for kw in info["relevant_for"] if kw in query_lower)
            if kw_score > best_score:
                best_score = kw_score
                best_tool = tool_name

        # Check if best tool was used
        tools_used = [s.name for s in tool_steps]
        used_correct = best_tool in tools_used if best_tool else True

        score = 0.0
        reasons = []

        # 1. Correct tool used? (0.5)
        if used_correct:
            score += 0.50
            reasons.append(f"Correct tool '{best_tool}' was used")
        else:
            reasons.append(f"Best tool '{best_tool}' was NOT used (used: {tools_used})")

        # 2. Efficiency - not too many tool calls (0.3)
        n_calls = len(tool_steps)
        if n_calls == 1:
            score += 0.30
            reasons.append("Efficient: 1 tool call")
        elif n_calls == 2:
            score += 0.15
            reasons.append(f"Slightly wasteful: {n_calls} tool calls")
        else:
            reasons.append(f"Wasteful: {n_calls} tool calls (should be 1-2)")

        # 3. No irrelevant tools (0.2)
        irrelevant = [t for t in tools_used if t != best_tool]
        if not irrelevant:
            score += 0.20
            reasons.append("No irrelevant tools called")
        else:
            reasons.append(f"Irrelevant tools called: {irrelevant}")

        return EvalResult(
            evaluator_name=self.name,
            score=round(min(score, 1.0), 2),
            reasoning="; ".join(reasons),
            details={"tool_usage": round(score, 2), "tools_used": tools_used, "best_tool": best_tool},
        )


class ReasoningEvaluator(BaseEvaluator):
    """Scores whether the agent reasoned before acting."""
    name = "reasoning"

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        reasoning_steps = [s for s in trace.steps if s.step_type == StepType.REASONING]
        tool_steps = [s for s in trace.steps if s.step_type == StepType.TOOL_CALL]

        if not reasoning_steps:
            # No reasoning at all
            if tool_steps:
                return EvalResult(
                    evaluator_name=self.name, score=0.30,
                    reasoning="Agent acted without thinking - no reasoning step before tool calls",
                    details={"reasoning": 0.30},
                )
            return EvalResult(
                evaluator_name=self.name, score=0.40,
                reasoning="No reasoning or tool steps found",
                details={"reasoning": 0.40},
            )

        # Check reasoning quality
        reasoning_text = reasoning_steps[0].output_data.get("reasoning", "")
        score = 0.5  # Base for having reasoning

        reasons = ["Has reasoning step"]

        # Does it mention the query?
        if trace.query and trace.query.lower()[:20] in reasoning_text.lower():
            score += 0.15
            reasons.append("References the query")

        # Does it mention tools?
        if "tool" in reasoning_text.lower() or any(t in reasoning_text.lower() for t in TOOLS):
            score += 0.15
            reasons.append("Considers available tools")

        # Does it explain why?
        if "because" in reasoning_text.lower() or "best" in reasoning_text.lower():
            score += 0.20
            reasons.append("Explains tool selection rationale")

        return EvalResult(
            evaluator_name=self.name,
            score=round(min(score, 1.0), 2),
            reasoning="; ".join(reasons),
            details={"reasoning": round(min(score, 1.0), 2)},
        )


class AnswerGroundingEvaluator(BaseEvaluator):
    """Scores whether the answer is grounded in tool outputs (no hallucination)."""
    name = "grounding"

    HALLUCINATION_MARKERS = [
        "quantum entanglement", "blockchain", "cryptocurrency",
        "300%", "fastest growing", "recent studies",
    ]

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        response = (trace.response or "").lower()

        score = 0.0
        reasons = []

        # 1. No hallucinations (0.50)
        hallucinations = [m for m in self.HALLUCINATION_MARKERS if m in response]
        if not hallucinations:
            score += 0.50
            reasons.append("No hallucinations detected")
        else:
            reasons.append(f"HALLUCINATION: '{hallucinations[0]}'")

        # 2. Response uses tool output (0.30)
        tool_steps = [s for s in trace.steps if s.step_type == StepType.TOOL_CALL]
        if tool_steps:
            tool_outputs = " ".join(
                s.output_data.get("output", "").lower() for s in tool_steps
            )
            # Check if key words from tool output appear in response
            tool_words = [w for w in tool_outputs.split() if len(w) > 4][:15]
            overlap = sum(1 for w in tool_words if w in response)
            grounding_ratio = overlap / max(len(tool_words), 1)
            grounding_score = min(grounding_ratio * 0.60, 0.30)
            score += grounding_score
            reasons.append(f"Grounding: {overlap}/{len(tool_words)} tool output words used")
        else:
            reasons.append("No tool outputs to ground against")

        # 3. Confidence and clarity (0.20)
        if "i don't know" not in response and "not sure" not in response:
            score += 0.20
            reasons.append("Confident response")
        else:
            score += 0.05
            reasons.append("Uncertain response")

        return EvalResult(
            evaluator_name=self.name,
            score=round(min(score, 1.0), 2),
            reasoning="; ".join(reasons),
            details={"correctness": round(min(score, 1.0), 2)},
        )


# =============================================================================
# 4. AGENT OPTIMIZER
# =============================================================================

class AgentOptimizer(BaseOptimizer):
    """Optimizer specialized for ReAct agents - fixes prompt, tools, temp, reasoning."""
    name = "agent_optimizer"

    def suggest(self, traces: list[ExecutionTrace], current_config: OptimizationConfig) -> list[Suggestion]:
        if not traces:
            return []

        suggestions = []

        # Aggregate scores
        score_lists: dict[str, list[float]] = {}
        for trace in traces:
            for r in trace.eval_results:
                score_lists.setdefault(r.evaluator_name, []).append(r.score)
                for key, val in r.details.items():
                    if isinstance(val, (int, float)):
                        score_lists.setdefault(key, []).append(float(val))

        avg = {k: sum(v) / len(v) for k, v in score_lists.items()}

        # Rule 1: Poor tool usage -> reduce enabled tools + lower temperature
        tool_score = avg.get("tool_usage", 1.0)
        if tool_score < 0.7:
            # Find which tools are actually being used correctly
            used_tools = set()
            for trace in traces:
                for step in trace.steps:
                    if step.step_type == StepType.TOOL_CALL:
                        used_tools.add(step.name)

            # Suggest limiting to commonly needed tools
            core_tools = ["web_search", "calculator", "weather_api"]
            current_tools = current_config.enabled_tools or list(TOOLS.keys())
            if len(current_tools) > 3:
                suggestions.append(Suggestion(
                    param_name="enabled_tools",
                    old_value=current_tools,
                    new_value=core_tools,
                    reasoning=f"Tool usage score is {tool_score:.2f}. Agent is confused by {len(current_tools)} tools and calls irrelevant ones. Limiting to {len(core_tools)} core tools will reduce confusion.",
                    confidence=0.80,
                    category="agent",
                ))

            # Also limit max tool calls
            current_max = current_config.max_tool_calls or 5
            if current_max > 2:
                suggestions.append(Suggestion(
                    param_name="max_tool_calls",
                    old_value=current_max,
                    new_value=2,
                    reasoning=f"Agent makes up to {current_max} tool calls per query. Limiting to 2 prevents wasteful calls.",
                    confidence=0.75,
                    category="agent",
                ))

        # Rule 2: No reasoning -> enable Chain-of-Thought
        reasoning_score = avg.get("reasoning", 1.0)
        if reasoning_score < 0.5:
            current_strategy = current_config.reasoning_strategy or "default"
            if current_strategy != "cot":
                suggestions.append(Suggestion(
                    param_name="reasoning_strategy",
                    old_value=current_strategy,
                    new_value="cot",
                    reasoning=f"Reasoning score is {reasoning_score:.2f}. Agent acts without thinking. Chain-of-Thought forces the agent to analyze the query and select the right tool BEFORE acting.",
                    confidence=0.90,
                    category="agent",
                ))

        # Rule 3: Hallucinations -> lower temperature + improve prompt
        grounding_score = avg.get("correctness", avg.get("grounding", 1.0))
        if grounding_score < 0.7:
            current_temp = current_config.temperature if current_config.temperature is not None else 0.9
            if current_temp > 0.3:
                suggestions.append(Suggestion(
                    param_name="temperature",
                    old_value=current_temp,
                    new_value=0.1,
                    reasoning=f"Grounding score is {grounding_score:.2f}. Temperature {current_temp} causes hallucinations. Setting to 0.1 for deterministic, factual outputs.",
                    confidence=0.90,
                    category="agent",
                ))

        # Rule 4: Improve system prompt
        quality_score = avg.get("grounding", avg.get("correctness", 1.0))
        current_prompt = current_config.system_prompt or "vague"
        if quality_score < 0.8 and "structured" not in (current_prompt or ""):
            suggestions.append(Suggestion(
                param_name="system_prompt",
                old_value="vague (generic assistant prompt)",
                new_value="structured",
                reasoning=f"Current prompt is too vague, leading to unfocused responses. A structured prompt with explicit rules (THINK -> ACT -> OBSERVE -> RESPOND) improves tool selection and answer quality.",
                confidence=0.85,
                category="prompt",
            ))

        return suggestions


# =============================================================================
# 5. THE DEMO
# =============================================================================

def print_header(text: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def print_config(config: OptimizationConfig, indent: str = "    ") -> None:
    prompt_label = "vague" if not config.system_prompt or config.system_prompt == "vague" else "structured"
    tools = config.enabled_tools or list(TOOLS.keys())
    print(f"{indent}temperature   = {config.temperature}")
    print(f"{indent}system_prompt = {prompt_label}")
    print(f"{indent}reasoning     = {config.reasoning_strategy or 'none'}")
    print(f"{indent}max_tools     = {config.max_tool_calls or 'unlimited'}")
    print(f"{indent}enabled_tools = {tools}")


def run_eval_round(wrapped: Retuner, queries: list[str], label: str) -> dict[str, float]:
    print(f"\n  --- {label} ---")
    all_scores: dict[str, list[float]] = {}

    for q in queries:
        response = wrapped.run(q)
        scores_str = ", ".join(f"{r.evaluator_name}={r.score:.2f}" for r in response.eval_results)

        # Show tool calls
        tool_steps = [s for s in (response.trace.steps if response.trace else [])
                      if s.step_type == StepType.TOOL_CALL]
        tools_used = [s.name for s in tool_steps]

        reasoning_steps = [s for s in (response.trace.steps if response.trace else [])
                           if s.step_type == StepType.REASONING]

        answer = (response.output or "")[:85]
        if len(response.output or "") > 85:
            answer += "..."

        print(f"    Q: {q}")
        if reasoning_steps:
            print(f"       Think: (chain-of-thought reasoning)")
        print(f"       Tools: {tools_used}")
        print(f"       A: {answer}")
        print(f"       Scores: [{scores_str}]")

        for r in response.eval_results:
            all_scores.setdefault(r.evaluator_name, []).append(r.score)

    return {k: sum(v) / len(v) for k, v in all_scores.items()}


def print_scoreboard(scores: dict[str, float], baseline: dict[str, float] | None = None) -> None:
    for name, avg in scores.items():
        bar_len = int(avg * 30)
        bar = "#" * bar_len + " " * (30 - bar_len)
        if baseline and name in baseline:
            delta = avg - baseline[name]
            sign = "+" if delta > 0 else ""
            label = f"({sign}{delta:.2f})"
        else:
            label = "GOOD" if avg >= 0.7 else "NEEDS WORK" if avg >= 0.4 else "POOR"
        print(f"     {name:20s} [{bar}] {avg:.2f}  {label}")


def main():
    random.seed(42)

    # Test queries covering different tool needs
    queries = [
        "What is machine learning and how does it work?",
        "What's the weather like in Paris today?",
        "Calculate 15% of 200",
        "Explain how ReAct agents use tools",
        "What's the company Q3 revenue?",
    ]

    # Storage setup
    import tempfile, os
    db_path = os.path.join(tempfile.mkdtemp(), "agent_demo.db")
    from retune.storage.sqlite_storage import SQLiteStorage

    # =====================================================
    # PHASE 1: BAD CONFIG
    # =====================================================
    print_header("PHASE 1: Agent with DELIBERATELY BAD config")

    bad_config = OptimizationConfig(
        temperature=0.9,                           # Too random!
        system_prompt="vague",                     # Too vague!
        reasoning_strategy=None,                   # No CoT!
        max_tool_calls=5,                          # Too many!
        enabled_tools=list(TOOLS.keys()),          # All 6 tools = confusion!
    )
    print("  Starting config (bad):")
    print_config(bad_config)

    adapter = SimulatedReActAdapter(agent=lambda q: q)

    wrapped = Retuner(
        agent=lambda q: q,
        adapter=adapter,
        mode=Mode.OBSERVE,
        config=bad_config,
        evaluators=[
            ToolUsageEvaluator(),
            ReasoningEvaluator(),
            AnswerGroundingEvaluator(),
        ],
        optimizer=AgentOptimizer(),
        storage=SQLiteStorage(db_path),
    )

    # Observe
    for q in queries:
        wrapped.run(q)
    print(f"\n  Traces collected: {len(wrapped.get_traces())}")

    # =====================================================
    # PHASE 2: EVALUATE
    # =====================================================
    print_header("PHASE 2: EVALUATE - What's wrong with this agent?")
    wrapped.set_mode(Mode.EVALUATE)

    baseline_scores = run_eval_round(wrapped, queries, "Baseline Evaluation")

    print(f"\n  >> BASELINE SCORES:")
    print_scoreboard(baseline_scores)

    # =====================================================
    # PHASE 3: IMPROVE
    # =====================================================
    print_header("PHASE 3: IMPROVE - Self-diagnosing and fixing")
    wrapped.set_mode(Mode.IMPROVE)
    wrapped._auto_improve = True

    print(f"\n  Config BEFORE (v{wrapped.version}):")
    print_config(wrapped.get_config())

    # Trigger improvement
    response = wrapped.run("What is machine learning?")

    if response.suggestions:
        print(f"\n  >> {len(response.suggestions)} IMPROVEMENTS IDENTIFIED:")
        for i, s in enumerate(response.suggestions, 1):
            conf_bar = "#" * int(s.confidence * 10) + " " * (10 - int(s.confidence * 10))
            old_display = s.old_value
            new_display = s.new_value
            if isinstance(old_display, list):
                old_display = f"[{len(old_display)} tools]"
            if isinstance(new_display, list):
                new_display = f"[{', '.join(new_display)}]"

            print(f"\n  {i}. [{s.category.upper():>6s}] {s.param_name}")
            print(f"     Before: {old_display}")
            print(f"     After:  {new_display}")
            print(f"     Confidence: [{conf_bar}] {s.confidence:.0%}")
            print(f"     Why: {s.reasoning}")

    print(f"\n  Config AFTER auto-improvement (v{wrapped.version}):")
    print_config(wrapped.get_config())

    # =====================================================
    # PHASE 4: RE-EVALUATE
    # =====================================================
    print_header("PHASE 4: RE-EVALUATE - Is the agent better now?")
    wrapped.set_mode(Mode.EVALUATE)

    improved_scores = run_eval_round(wrapped, queries, "Post-Improvement Evaluation")

    print(f"\n  >> IMPROVED SCORES:")
    print_scoreboard(improved_scores, baseline=baseline_scores)

    # =====================================================
    # FINAL SUMMARY
    # =====================================================
    print_header("FINAL SUMMARY: Agent Self-Improvement Results")

    print(f"\n  {'Metric':<20s} {'Before':>8s} {'After':>8s} {'Change':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

    total_b, total_a, count = 0, 0, 0
    for name in baseline_scores:
        b = baseline_scores[name]
        a = improved_scores.get(name, b)
        d = a - b
        sign = "+" if d > 0 else ""
        marker = " << IMPROVED" if d > 0.05 else ""
        print(f"  {name:<20s} {b:>8.2f} {a:>8.2f} {sign}{d:>7.2f}{marker}")
        total_b += b; total_a += a; count += 1

    if count:
        avg_b, avg_a = total_b / count, total_a / count
        d = avg_a - avg_b
        sign = "+" if d > 0 else ""
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
        print(f"  {'OVERALL':<20s} {avg_b:>8.2f} {avg_a:>8.2f} {sign}{d:>7.2f}")

    # What changed
    history = wrapped.get_improvement_history()
    if history:
        print(f"\n  Changes applied automatically (v1 -> v{wrapped.version}):")
        for entry in history:
            for s in entry["suggestions"]:
                old = s['old_value']
                new = s['new_value']
                if isinstance(old, list):
                    old = f"[{len(old)} tools]"
                if isinstance(new, list):
                    new = f"[{', '.join(new)}]"
                print(f"    {s['param_name']}: {old} -> {new}")

    print(f"\n  The agent diagnosed its own weaknesses and fixed them!")
    print(f"  - Reduced tool confusion by limiting available tools")
    print(f"  - Added chain-of-thought reasoning before tool calls")
    print(f"  - Lowered temperature to prevent hallucinations")
    print(f"  - Improved system prompt for structured behavior")
    print()


if __name__ == "__main__":
    main()
