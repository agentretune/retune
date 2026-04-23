# retune

**Retune your AI agents** -- self-improving evaluation & optimization framework.

> Make any LLM agent, RAG pipeline, or workflow self-improving in production.

[![PyPI version](https://img.shields.io/pypi/v/retune)](https://pypi.org/project/retune/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/agentretune/retune/actions/workflows/test.yml/badge.svg)](https://github.com/agentretune/retune/actions/workflows/test.yml)

[Website](https://agentretune.com) | [Documentation](https://agentretune.com/docs) | [GitHub](https://github.com/agentretune/retune)

---

## What is retune?

`retune` is a framework-agnostic SDK that wraps any LLM-based system and makes it self-improving through automated observation, evaluation, and optimization. It works with LangChain, LangGraph, or any custom pipeline.

## Installation

```bash
pip install retune
```

| Extra | Command | What you get |
|-------|---------|--------------|
| LangChain | `pip install retune[langchain]` | LangChain adapter |
| LangGraph | `pip install retune[langgraph]` | LangGraph adapter |
| LLM Judge | `pip install retune[llm]` | OpenAI-powered evaluator |
| Anthropic | `pip install retune[anthropic]` | Claude models |
| Google | `pip install retune[google]` | Gemini models |
| Ollama | `pip install retune[ollama]` | Local models |
| Server | `pip install retune[server]` | FastAPI dashboard |
| Everything | `pip install retune[all]` | All of the above |
| Dev | `pip install retune[dev]` | pytest, ruff, mypy |

## Feature Availability

| Feature | Availability |
|---|---|
| Observability (tracing) | OSS, free, local-first |
| Evaluation (all evaluators incl. LLM judge) | OSS, free, BYO-key |
| Optimization (prompts + tools + RAG) | Cloud, 15 free trial runs, then Pro/Team/Enterprise |
| Local dashboard | OSS (`retune dashboard`) |
| Hosted dashboard | agentretune.com (Pro+) |

## Quickstart

```python
from retune import Retuner, Mode

# Your existing agent (any callable)
def my_agent(query: str) -> str:
    return call_llm(query)

# Wrap it
retuner = Retuner(
    agent=my_agent,
    adapter="langchain",
    mode=Mode.OBSERVE,
    api_key="rt-...",           # optional; required only for optimize()
    agent_purpose="...",        # required when mode=Mode.IMPROVE
)

# Run your agent normally — retuner captures traces
response = retuner.run("What is ML?")

# When you want to optimize
retuner.set_mode(Mode.IMPROVE)
report = retuner.optimize(
    source="last_n_traces", n=20,
    axes=["prompt", "tools", "rag"],
    rewriter_llm="claude-3-7-sonnet",
)
report.show()
retuner.apply_report(report, tier=1)  # apply one-click suggestions
```

## Framework Adapters

### Custom (any callable)

```python
from retune import Retuner, Mode

retuner = Retuner(agent=my_fn, adapter="custom", mode=Mode.OBSERVE)
response = retuner.run("Hello")
```

### LangChain

```python
from retune import Retuner, Mode

chain = prompt | llm | parser  # your LangChain chain

retuner = Retuner(
    agent=chain,
    adapter="langchain",
    mode=Mode.EVALUATE,
    evaluators=["llm_judge", "retrieval", "latency"],
)

response = retuner.run("What is RAG?")
print(response.eval_results)  # quality scores
```

### LangGraph

```python
from retune import Retuner, Mode

compiled_graph = graph.compile()

retuner = Retuner(
    agent=compiled_graph,
    adapter="langgraph",
    mode=Mode.OBSERVE,
)

response = retuner.run("Plan a trip to Paris")
for step in response.trace.steps:
    print(f"[{step.step_type}] {step.name}")
```

## The Fan Regulator Model

Control how much post-processing happens after each run:

| Mode | What it does | Overhead |
|------|-------------|----------|
| `OFF` | Pure passthrough | None |
| `OBSERVE` | Capture execution traces | Low |
| `EVALUATE` | + Run evaluators, score quality | Medium |
| `IMPROVE` | + Generate optimization suggestions | High |

```python
retuner.set_mode(Mode.IMPROVE)  # Turn up when tuning
# ... run evaluation dataset ...
retuner.set_mode(Mode.OFF)      # Turn off for production
```

## Accept/Reject Optimization Flow

When in `IMPROVE` mode, retune generates suggestions that you can review:

```python
response = retuner.run("test query")
for suggestion in response.suggestions:
    print(f"{suggestion.param_name}: {suggestion.old_value} -> {suggestion.new_value}")
    # Accept or reject each suggestion
    retuner.accept_suggestion(suggestion)
    # or: retuner.reject_suggestion(suggestion)
```

## Beam Search APO (Automatic Prompt Optimization)

retune includes a beam-search-based prompt optimizer that iteratively rewrites your system prompt:

```python
from retune import Retuner, BeamSearchConfig

config = BeamSearchConfig(
    beam_width=2,         # candidates kept per round
    branch_factor=2,      # rewrites per candidate
    rounds=3,             # search depth
    max_rollout_queries=5,
)

retuner = Retuner(agent=my_agent, adapter="custom", mode=Mode.IMPROVE)
best_prompt = retuner.optimize_prompt(
    initial_prompt="You are a helpful assistant.",
    eval_dataset=dataset,
    beam_config=config,
)
```

## Multi-Provider LLM Support

retune works with any LLM provider through LangChain integrations:

```python
from retune import set_default_llm

# OpenAI (default)
set_default_llm("openai", model="gpt-4o")

# Anthropic
set_default_llm("anthropic", model="claude-sonnet-4-20250514")

# Google
set_default_llm("google", model="gemini-pro")

# Local via Ollama
set_default_llm("ollama", model="llama3")
```

## Evaluators

| Evaluator | What it scores | Install |
|-----------|---------------|---------|
| `llm_judge` | Correctness, completeness, relevance | `retune[llm]` |
| `retrieval` | Document retrieval quality | Built-in |
| `latency` | Execution speed | Built-in |
| `cost` | Token usage efficiency | Built-in |

## Self-Improvement Loop

```python
# 1. Collect traces
retuner.set_mode(Mode.OBSERVE)
for query in queries:
    retuner.run(query)

# 2. Evaluate
retuner.set_mode(Mode.EVALUATE)
summary = retuner.get_eval_summary()

# 3. Get improvement suggestions
retuner.set_mode(Mode.IMPROVE)
response = retuner.run("test query")
for s in response.suggestions:
    print(f"{s.param_name}: {s.old_value} -> {s.new_value}")

# 4. Apply and run in production
retuner.set_mode(Mode.OFF)
```

## Architecture

```
Your Agent/RAG (LangGraph, LangChain, custom)
        |
   Adapter Layer  -- framework-specific -> universal trace
        |
   Execution Trace  -- standard format for all frameworks
        |
   Evaluation Engine  -- modular scorers (LLM judge, retrieval, latency, cost)
        |
   Optimization Engine  -- beam search APO, rule-based suggestions
        |
   Accept/Reject  -- human-in-the-loop or auto-apply
        |
   Improved Config  -- better prompts, parameters, retrieval
```

## Environment Variables

```bash
# LLM provider keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# retune config (all optional, sensible defaults)
RETUNE_STORAGE_PATH=./retune.db
RETUNE_DEFAULT_MODE=observe
RETUNE_LOG_LEVEL=INFO
RETUNE_EVAL_LLM_MODEL=gpt-4o-mini
```

## License

MIT -- see [LICENSE](LICENSE).
