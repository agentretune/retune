# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.3.0 - 2026-04-13

### Major changes — open-core repositioning

- **Observability and evaluation are now fully open source.** All evaluators
  (including `LLMJudgeEvaluator` and `PairwiseJudgeEvaluator`) run locally on
  the user's LLM API key. No cloud dependency for observe/evaluate.
- **Cloud-hosted Optimization is the paid pillar.** 15 free trial runs with a
  cloud API key; Pro ($49), Team ($99), Enterprise plans for unlimited.

### Optimizer

- New `Retuner.optimize(source, n, axes, rewriter_llm, guardrails)` method
  triggers a cloud-side optimization run. Returns an `OptimizationReport` with
  tiered apply-manifest (Tier 1 one-click apply, Tier 2 copy-paste snippets,
  Tier 3 conceptual suggestions).
- Three subagents dispatched by the `OptimizerOrchestrator`:
  - **PromptOptimizerAgent** — Beam Search APO with multi-round pruning, LLM
    proposed rewrites (rewriter_llm user's choice: OpenAI, Anthropic, Google),
    few-shot curation from high-scoring traces.
  - **ToolOptimizerAgent** — drops unused tools, rewrites descriptions via
    LLM, tightens arg schemas, proposes new tools (Tier 3).
  - **RAGOptimizerAgent** — parameter sweeps over retrieval_k, chunk_size,
    chunk_overlap, reranker, retrieval strategy.
- Judge + guardrails scoring (LLM judge primary, cost/latency guardrails);
  declarative JSON reward specs for advanced users.
- Pareto-frontier visualization in the cloud dashboard.
- Feedback loop: accepted/rejected suggestions feed context of subsequent
  optimization runs for the same org.

### SDK

- `Retuner.__init__` now requires `agent_purpose="..."` when `mode=Mode.IMPROVE`.
- `retune dashboard` CLI now serves a pure-SDK local app reading local SQLite.
  The previous cloud-backed dashboard moved to agentretune.com.
- Tool introspection: SDK reads `agent.tools` / `bind_tools` / LangGraph tool
  nodes at `optimize()` time and uploads metadata alongside traces.
- Adapter `apply_retrieval_override(**kwargs)` hook — LangChain adapter mutates
  `retriever.search_kwargs`; other adapters get no-op default.

### Cloud (private repo)

- New `retune-cloud` FastAPI service with routes under `/api/v1/optimize/*`,
  `/api/v1/jobs/*`, `/api/v1/billing/*`, `/api/v1/optimize/{run_id}/feedback`,
  `/api/v1/optimize/runs`, `/api/v1/optimize/{run_id}/pareto`.
- Stripe webhook completes Pro/Team/Enterprise upgrades end-to-end.
- DB schema: `optimization_runs`, `optimization_candidates`,
  `optimization_reports`, `optimization_run_traces`, `optimization_run_tools`,
  `optimization_run_retrieval_config`, `optimization_feedback`.

### Breaking changes

- `agent_purpose=` required for `Mode.IMPROVE`.
- Stale outer `server/` directory removed (was a monorepo leftover).
- `OptimizerDeepAgent` (v0.2 SDK wrapper) code is unchanged but no longer the
  recommended entry point — use `Retuner.optimize(...)` instead.

## [0.1.0] - 2026-03-21

### Added

- Core `Retuner` wrapper with fan-regulator modes (OFF, OBSERVE, EVALUATE, IMPROVE)
- Framework adapters: Custom, LangChain, LangGraph
- Modular evaluators: LLM Judge, Retrieval Quality, Latency, Cost
- Beam Search APO (Automatic Prompt Optimization) agent
- Accept/reject flow for optimization suggestions
- Multi-provider LLM support (OpenAI, Anthropic, Google, Ollama)
- Universal execution trace format
- SQLite-backed storage for traces and evaluation results
- Tool registry for agent tool management
- Pydantic-based configuration via environment variables
- PEP 561 type checking support (py.typed)
- 105 tests with full coverage of core functionality

[0.1.0]: https://github.com/agentretune/retune/releases/tag/v0.1.0
