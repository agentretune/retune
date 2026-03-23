"""
Real-world demo: LangGraph ReAct Agent + Retune
================================================
Builds a ReAct agent with 3 tools (calculator, weather, search),
then runs it through ALL retune modes to prove everything works
with a real LLM (Claude).

Usage:
    python examples/demo_langgraph_react.py
"""

import os
import math
import json
import logging
from datetime import datetime

# --- Set API key ---
os.environ.setdefault("ANTHROPIC_API_KEY", "your-api-key-here")

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from retune import Retuner, Mode

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ============================================================
# 1. Define real tools
# ============================================================

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Supports +, -, *, /, sqrt, pow, pi, etc.

    Args:
        expression: A math expression like '2 + 2' or 'sqrt(144)' or 'pow(2, 10)'
    """
    allowed = {
        "sqrt": math.sqrt, "pow": pow, "abs": abs,
        "pi": math.pi, "e": math.e, "log": math.log,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. 'London' or 'Tokyo'
    """
    # Simulated weather data (in a real app, this would call an API)
    weather_data = {
        "london": {"temp": 12, "condition": "Cloudy", "humidity": 78},
        "tokyo": {"temp": 22, "condition": "Sunny", "humidity": 55},
        "new york": {"temp": 18, "condition": "Partly cloudy", "humidity": 62},
        "paris": {"temp": 15, "condition": "Rainy", "humidity": 85},
        "dhaka": {"temp": 32, "condition": "Hot and humid", "humidity": 90},
    }
    key = city.lower()
    if key in weather_data:
        w = weather_data[key]
        return f"Weather in {city}: {w['temp']}°C, {w['condition']}, Humidity: {w['humidity']}%"
    return f"Weather in {city}: 25°C, Clear skies, Humidity: 60% (default estimate)"


@tool
def search_knowledge(query: str) -> str:
    """Search a knowledge base for information. Returns relevant facts.

    Args:
        query: The search query
    """
    # Simulated knowledge base
    kb = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes readability and supports multiple paradigms.",
        "machine learning": "Machine learning is a subset of AI where systems learn patterns from data. Key types: supervised, unsupervised, and reinforcement learning.",
        "langchain": "LangChain is a framework for building LLM-powered applications. It provides chains, agents, and retrieval components.",
        "retune": "Retune is a framework-agnostic SDK that wraps any LLM system and makes it self-improving through observation, evaluation, and optimization.",
        "rag": "Retrieval-Augmented Generation (RAG) combines a retriever with a generator. The retriever fetches relevant documents, and the LLM generates answers grounded in them.",
    }
    query_lower = query.lower()
    results = []
    for topic, info in kb.items():
        if topic in query_lower or any(w in query_lower for w in topic.split()):
            results.append(info)
    if results:
        return "Found: " + " | ".join(results)
    return f"No specific results found for '{query}'. Try a more specific search term."


# ============================================================
# 2. Build LangGraph ReAct Agent
# ============================================================

tools = [calculator, get_weather, search_knowledge]
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0).bind_tools(tools)


def agent_node(state: MessagesState):
    """The agent decides whether to use tools or respond directly."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState) -> str:
    """Route: if the last message has tool calls, go to tools. Otherwise, end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# Build the graph
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

react_agent = graph.compile()


# ============================================================
# 3. Run through ALL Retune modes
# ============================================================

def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    # --- MODE 1: OFF (pure passthrough) ---
    separator("MODE: OFF -- Pure passthrough, zero overhead")

    retuner = Retuner(
        agent=react_agent,
        adapter="langgraph",
        mode=Mode.OFF,
        evaluators=["latency", "cost", "retrieval"],
    )

    resp = retuner.run("What is 2 + 2?")
    print(f"Output: {resp.output[:200]}")
    print(f"Trace: {resp.trace}")  # None in OFF mode
    print(f"Eval results: {resp.eval_results}")  # Empty in OFF mode

    # --- MODE 2: OBSERVE (captures traces) ---
    separator("MODE: OBSERVE -- Capture execution traces")
    retuner.set_mode(Mode.OBSERVE)

    queries = [
        "What's the weather in Tokyo and how does it compare to London?",
        "Calculate sqrt(144) + pow(2, 10)",
        "Search for information about machine learning and RAG",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        resp = retuner.run(q)
        print(f"Output: {resp.output[:200]}")
        print(f"Trace ID: {resp.trace.trace_id}")
        print(f"Steps captured: {len(resp.trace.steps)}")
        for step in resp.trace.steps:
            duration = (step.ended_at - step.started_at).total_seconds() * 1000
            print(f"  [{step.step_type.value}] {step.name} ({duration:.0f}ms)")

    # --- MODE 3: EVALUATE (captures + scores) ---
    separator("MODE: EVALUATE -- Capture + run evaluators")
    retuner.set_mode(Mode.EVALUATE)

    eval_queries = [
        "What is the weather in Paris? Also tell me what Python is.",
        "Calculate pow(3, 4) and then search for info about LangChain",
    ]

    for q in eval_queries:
        print(f"\nQuery: {q}")
        resp = retuner.run(q)
        print(f"Output: {resp.output[:200]}")
        print(f"\nEval Results:")
        for ev in resp.eval_results:
            print(f"  {ev.evaluator_name}: {ev.score:.3f} -- {ev.reasoning}")

    # --- Evaluation summary ---
    separator("EVALUATION SUMMARY -- Aggregated scores")
    summary = retuner.get_eval_summary()
    print(f"Total traces: {summary['total_traces']}")
    for name, stats in summary.get("scores", {}).items():
        print(f"  {name}: mean={stats['mean']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}")

    # --- MODE 4: IMPROVE (captures + scores + suggestions) ---
    separator("MODE: IMPROVE -- Generate optimization suggestions")
    retuner.set_mode(Mode.IMPROVE)

    resp = retuner.run("What is retune and how does it work?")
    print(f"Output: {resp.output[:200]}")

    print(f"\nSuggestions generated: {len(resp.suggestions)}")
    for s in resp.suggestions:
        print(f"  [{s.category}] {s.param_name}: {s.old_value} -> {s.new_value}")
        print(f"    Reason: {s.reasoning}")
        print(f"    Confidence: {s.confidence:.0%}")
        print(f"    Status: {s.status.value}")

    # --- Accept/Reject flow ---
    separator("ACCEPT/REJECT FLOW -- Human-in-the-loop")

    pending = retuner.get_pending_suggestions()
    print(f"Pending suggestions: {len(pending)}")

    if pending:
        # Accept the first suggestion
        first = pending[0]
        print(f"\nAccepting: {first.param_name} ({first.old_value} -> {first.new_value})")
        retuner.accept_suggestion(first.suggestion_id)
        print(f"Config after accept: top_k={retuner.get_config().top_k}, "
              f"temperature={retuner.get_config().temperature}")

        if len(pending) > 1:
            second = pending[1]
            print(f"\nRejecting: {second.param_name} ({second.old_value} -> {second.new_value})")
            retuner.reject_suggestion(second.suggestion_id)

    # --- Show improvement history ---
    separator("IMPROVEMENT HISTORY")
    history = retuner.get_improvement_history()
    for entry in history:
        print(f"  v{entry['version']}: {entry['action']} -- {entry.get('param_name', 'all')}")

    # --- Run evaluation dataset ---
    separator("EVALUATION DATASET -- Batch evaluation")
    dataset = [
        {"query": "What is the weather in Dhaka?"},
        {"query": "Calculate sqrt(256)"},
        {"query": "Search for information about Python"},
    ]

    results = retuner.run_evaluation_dataset(dataset)
    print(f"Evaluated {results['total_queries']} queries")
    print(f"Aggregate scores:")
    for name, score in results["aggregate_scores"].items():
        print(f"  {name}: {score:.3f}")

    # --- Revert all ---
    separator("REVERT -- Undo all changes")
    print(f"Config before revert: top_k={retuner.get_config().top_k}")
    retuner.revert_all()
    print(f"Config after revert: top_k={retuner.get_config().top_k}")
    print(f"Final version: v{retuner.version}")

    separator("DONE -- All retune features demonstrated with a real LangGraph ReAct agent!")


if __name__ == "__main__":
    main()
