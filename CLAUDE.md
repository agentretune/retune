# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`retune` — a framework-agnostic SDK that wraps any LLM agent / RAG system and makes it self-improving via observation, automated evaluation, and optimization. Published as the `retune` package from `src/retune/`. Python ≥3.10.

## Commands

Install for development (matches CI):

```bash
pip install -e ".[dev,langchain,langgraph,llm]"
```

Use `.[all]` to also pull in anthropic / google / ollama / server / embeddings extras.

Lint, type-check, test (CI runs these in order on Python 3.10 / 3.11 / 3.12):

```bash
ruff check src/ tests/
mypy src/retune/ --ignore-missing-imports
pytest tests/ -v --tb=short
```

Single test:

```bash
pytest tests/unit/test_wrapper.py::test_retuner_observe_mode -v
```

Run the CLI (entry point `retune = retune.__main__:main`):

```bash
python -m retune dashboard --port 8000 --db retune.db
python -m retune version
```

Ruff config: `line-length = 100`, rules `E, F, I, W` (see `[tool.ruff]` in `pyproject.toml`). `pytest` runs with `asyncio_mode = "auto"`.

## Architecture

### Data flow (cross-file, read these together)

```
user agent ──► Adapter ──► ExecutionTrace ──► Storage ──► Evaluators ──► Optimizer ──► Suggestions
                                                │                                         │
                                                └──► CloudClient (async thread)           └──► accept/reject/revert
```

Any framework (LangChain, LangGraph, plain callable) is normalized into a single `ExecutionTrace` (`core/`) carrying `query`, `response`, `steps[]`, `eval_results[]`, and `config_snapshot`. Adapters in `adapters/` are the translation layer — add a new framework by subclassing `BaseAdapter`.

### The four modes (central control knob)

`Mode` enum gates overhead at runtime and is the primary API surface users will toggle via `retuner.set_mode(...)`:

- `OFF` — passthrough, zero overhead
- `OBSERVE` — capture traces only
- `EVALUATE` — traces + run evaluators (scored 0–1)
- `IMPROVE` — evaluate + generate suggestions (optional auto-apply on HIGH confidence)

Suggestions are **PENDING by default**; user explicitly calls `accept_suggestion()` / `reject_suggestion()` / `revert_suggestion()`. Accepted suggestions mutate `_config` on the wrapper.

### Key modules

- `wrapper.py` — `Retuner` orchestrator, the main public entry point
- `auto_eval.py` — `AutoEvalController`: the RL loop. Tracks call count, compares recent mean vs. baseline, triggers optimization on drift (default threshold 0.1) or low score (<0.7)
- `usage_gate.py` — enforces 15 free deep-optimizer ops; unlimited for premium/cloud users. Consults `cloud/` with a 5-min cache
- `cloud/client.py` — background-thread queue that batches trace uploads to `https://api.agentretune.com`. Non-blocking — never put synchronous cloud calls on the hot path
- `storage/` — `SQLiteStorage` (local) and `CloudStorage` (inherits and mirrors to cloud)
- `agents/optimizer/` — `OptimizerDeepAgent` (LangGraph + Beam Search APO). `optimizers/basic.py` is the rule-based fallback. Selection: deep optimizer is used iff `use_deep_optimizer=True` or `beam_config` is supplied **and** LangGraph is importable — otherwise silently falls back to basic.
- `evaluators/` — pluggable; implement `BaseEvaluator.evaluate(trace) -> EvalResult`. Built-ins: `llm_judge`, `retrieval`, `latency`, `cost`

### Optional-dependency discipline

LangChain, LangGraph, and LLM providers are **optional extras**. Code that imports them must degrade gracefully when they're absent (the deep optimizer / basic optimizer split above is the canonical pattern). Don't promote an optional import to a hard dependency without adding it to `pyproject.toml`'s core `dependencies`.

### Cloud vs. local

No `RETUNE_API_KEY` ⇒ fully local, unlimited, SQLite only. With a key, `UsageGate` + `CloudClient` activate; traces sync to the cloud and usage is metered. Tests must not require network — mock `CloudClient` or run in local mode.

## Layout quick-reference

```
src/retune/          library code (see Architecture above)
tests/unit/          unit tests (wrapper, evaluators, optimizers, storage)
tests/integration/   end-to-end tests
examples/            runnable demos — quickstart.py is the minimal one
dashboard/           FastAPI backend + static HTML UI (served by `retune dashboard`)
deploy/              Docker / Fly / Render configs
```

See `MANUAL_STEPS.md` for production setup (Supabase, Stripe, JWT).
