# Retune Optimizer v0.3 — Design Spec

**Status:** Approved design, pre-plan
**Author:** Brainstorm between product owner and Claude
**Date:** 2026-04-12
**Target release:** v0.3.0 (3–4 months)

---

## 1. Context

Retune v0.2 ships two "deep agents" in a single SDK: one for evaluation, one for optimization, both paid-gated with a 15-free-operation trial. Adoption is bottlenecked by the paid gate on evaluation — teams want to evaluate their agents freely with their own LLM keys before committing to a paid product.

v0.3 is a **product repositioning** around an open-core model:

- **Observability + Evaluation → fully OSS.** All evaluators (including LLM-judge and pairwise-judge) ship in the SDK and run on the user's LLM key or a local model. Zero paid gates on evaluation.
- **Optimization → the paid pillar.** Rebuilt from a single deep agent into a comprehensive multi-agent optimization service that runs cloud-side and produces actionable improvement reports.

The bet: free evaluation drives SDK adoption → observed agents need improvement → optimization is the upsell. This mirrors the PostHog/Sentry open-core playbook.

This spec covers the **Optimizer** half of v0.3. The evaluation-to-OSS migration is a separate, smaller spec.

---

## 2. Scope

### In scope for v0.3

Three optimization axes:

| Axis | What can change |
|---|---|
| **Prompt** | System prompt rewrites (Beam Search APO), few-shot selection/ordering, task-instruction clarity |
| **Tools** | Tool selection (drop unused), tool description/schema rewriting, argument-schema tightening |
| **RAG** | Retrieval strategy (dense/sparse/hybrid), top-k, chunk size & overlap, reranker on/off & model, query rewriting |

### Out of scope for v0.3 (candidates for v0.4+)

- Agent graph structure changes (LangGraph node restructuring)
- Memory/state config optimization
- Model selection (switching the underlying LLM)
- Weight-level fine-tuning (never an axis — this is *agent* optimization, not *model* fine-tuning)

### Non-goals

- Modifying the user's source code directly. The optimizer never writes to user files. Its deliverable is a structured report.
- Running on-premise or air-gapped in v0.3. Cloud-only.
- Supporting users without an API key. Optimization is a cloud service; no local optimizer path.

---

## 3. Product positioning

The Retune Optimizer is a **cloud-hosted multi-agent optimization service** that comprehensively improves any LLM agent wrapped by the Retune SDK. It runs an orchestrated agent system across the three axes above, evaluates candidate configurations against real production traces, and returns a tiered apply-manifest — some changes are one-click, some are copy-paste code, some are conceptual engineering guidance.

It is not reinforcement learning on LLM weights. It is systematic search + evaluation over agent configuration.

---

## 4. User journey

### SDK API (source of truth)

```python
from retune import Retuner, Mode

retuner = Retuner(
    agent=my_agent,
    adapter="langgraph",
    mode=Mode.IMPROVE,
    api_key="rt-...",
    agent_purpose="Customer-support RAG bot for billing questions from our FAQ",
    success_criteria="Accurate, concise, cites the FAQ source when possible",
    dashboard_sync=True,   # opt-in cloud mirror for the hosted dashboard
)

# Trigger an optimization run — two sources:

# (a) Historical-replay over last N local traces
report = retuner.optimize(
    source="last_n_traces",
    n=50,
    axes="auto",                          # or ["prompt","tools","rag"]
    reward="judge_with_guardrails",       # default (C)
    rewriter_llm="claude-3-7-sonnet",     # user's choice for prompt rewrites
)

# (b) Forward-collect — optimize after the next N runs
job = retuner.optimize(source="collect_next", n=50, axes="auto")
report = job.wait()

# Review and apply
report.show()                        # tiered apply-manifest in console
report.apply(tier=1)                 # auto-applies SDK-reachable changes
report.copy_snippets(to="clipboard") # tier-2 code diffs for copy-paste
# tier-3 suggestions are read-only guidance
```

### Required inputs at wrapper init

- `agent_purpose: str` — one-line description. Required for IMPROVE mode. Fed to the LLM judge and to the Orchestrator's context. Surfaced back to the user in every report so they can correct misunderstandings.
- `success_criteria: str | None` — optional rubric guidance for the judge.
- `api_key: str` — required for IMPROVE mode.

### Dashboard-triggered runs

The cloud dashboard exposes the same API surface via a control-plane endpoint. A run initiated from the dashboard calls the SDK via long-poll (see §6); a run initiated from the SDK surfaces live in the dashboard. Both paths are first-class; neither is a wrapper around the other.

---

## 5. System architecture

```
┌─────────────────────── CLOUD OPTIMIZER ───────────────────────┐
│                                                                │
│                ┌───────────────────────┐                       │
│                │  OptimizerOrchestrator│ ← reads traces,       │
│                │   (planner, budget,   │   plans axes,         │
│                │    feedback memory)   │   routes subagents    │
│                └──┬─────────┬──────┬───┘                       │
│                   ▼         ▼      ▼                           │
│           ┌──────────┐ ┌────────┐ ┌──────────┐                 │
│           │ PromptOpt│ │ToolOpt │ │ RAGOpt   │                 │
│           │(Beam APO)│ │(select,│ │(k,chunk, │                 │
│           │          │ │ desc)  │ │ rerank)  │                 │
│           └─────┬────┘ └────┬───┘ └─────┬────┘                 │
│                 └────────────┼────────────┘                    │
│                              ▼                                 │
│                      ┌─────────────┐                           │
│                      │ JudgeAgent  │ ← reward: judge + guard   │
│                      └──────┬──────┘                           │
│                             ▼                                  │
│                   ┌───────────────────┐                        │
│                   │ ReportWriterAgent │ → tiered apply-manifest│
│                   └───────────────────┘                        │
└──────────────────────────────┬─────────────────────────────────┘
                               │ RunCandidate(config, queries)
                               ▼
┌────────────────────── USER'S SDK WORKER ──────────────────────┐
│ Executes user's agent with candidate config, user's LLM key.  │
│ Returns: trace + eval scores. No LLM keys or raw prompts      │
│ leave user infrastructure.                                    │
└────────────────────────────────────────────────────────────────┘
```

### Subagent responsibilities

- **OptimizerOrchestrator** — Analyzes uploaded traces for bottlenecks (tool-call failure rate, retrieval scores, latency percentiles, judge baseline). Decides which subagents to dispatch and in what order. Holds feedback memory from prior runs for this agent. Enforces the total wall-clock + token budget.

- **PromptOptimizerAgent** — Owns the prompt axis. Inherits Beam Search APO from the existing `OptimizerDeepAgent` (not rewritten). Adds LLM-proposed rewrites using `rewriter_llm` (user's choice). Also curates few-shot examples from high-scoring traces.

- **ToolOptimizerAgent** — Owns the tools axis. Tool discovery = introspection of the wrapped agent at build time (reads `agent.tools`, LangChain `BindTool` lists, LangGraph tool nodes). Never parses user source files. Proposes: drop unused tools, rewrite descriptions, tighten argument schemas. Tier-3 may suggest new tools. If the adapter doesn't expose tool metadata (e.g. opaque custom callable), the subagent falls back to trace-based inference (analyzing step names in `ExecutionTrace.steps[]`) and all suggestions drop to Tier-2/3.

- **RAGOptimizerAgent** — Owns the retrieval axis. Never modifies user code. For each candidate it proposes parameter variants; if the adapter exposes runtime override hooks (e.g. LangChain `RunnableConfig`, retriever config), the candidate runs with overrides applied. If the user's retriever is opaque, RAG suggestions drop to Tier-2 (copy-paste) or Tier-3 (conceptual) only.

- **JudgeAgent** — Scores every candidate. Default reward = judge rating × guardrails (§7). Uses declarative reward spec if user supplied one. Logs every evaluator dimension (not just scalar) to populate the Pareto plot.

- **ReportWriterAgent** — Assembles the final report. Surfaces the Orchestrator's understanding of the agent *first* so user can correct it. Renders Pareto data. Categorizes each suggestion into tiers 1/2/3 with confidence + estimated impact + evidence-from-trace citation.

---

## 6. SDK ↔ Cloud protocol

### Pattern

**Long-poll RPC.** Chosen over WebSocket for simplicity, NAT/firewall friendliness, and because candidate execution is seconds-latency (not real-time). The SDK polls `/v1/jobs/pending?run_id=...` every few seconds while a run is active.

### Messages

```
Cloud → SDK:  RunCandidate(run_id, candidate_id, config_overrides, query_set)
SDK → Cloud:  CandidateResult(run_id, candidate_id, trace, eval_scores)
Cloud → SDK:  JobComplete(run_id, report_url)
Cloud → SDK:  JobFailed(run_id, reason)   // triggers slot refund
```

### Privacy properties

- User's LLM API keys never leave the SDK.
- Raw prompt/response text leaves only if `dashboard_sync=True` OR is part of an active optimization-burst (and is purged after §8).
- Configuration metadata (prompt texts, tool definitions, retrieval params) *does* leave during optimization — this is unavoidable, since it's what the optimizer operates on. Enterprise users with prompt-text concerns can run the optimizer against synthetic trace datasets instead.

---

## 7. Reward signal

### Default: judge rating with guardrails

```
candidate_score = judge_rating                        # 0–10
  × (cost_ratio      ≤ cost_guardrail)                # else 0
  × (latency_ratio   ≤ latency_guardrail)             # else 0
```

Defaults: `cost ≤ 1.5× baseline`, `p95_latency ≤ 1.2× baseline`. User-configurable via `retuner.optimize(guardrails={...})`.

The judge rubric is derived from `agent_purpose` + `success_criteria`. The rubric itself is rendered in the report for audit.

### Custom reward — declarative only

User-supplied custom rewards are a **declarative JSON schema**, not raw Python. Arbitrary Python on the cloud optimizer is a security non-starter.

```json
{
  "objective": "maximize",
  "primary": {"evaluator": "llm_judge", "weight": 1.0},
  "penalties": [
    {"evaluator": "cost",    "threshold": "<= 0.002", "hard": true},
    {"evaluator": "latency", "threshold": "<= 3.0s",  "hard": false, "weight": 0.2}
  ],
  "extra_metrics": [{"evaluator": "retrieval_precision"}]
}
```

A future local-optimizer tier (not in v0.3) could support raw Python `reward_fn` since it would run in the user's process.

### Pareto visualization

JudgeAgent logs every evaluator dimension per candidate. The dashboard renders an interactive 3D scatter (quality × cost × latency) with the Pareto frontier highlighted. Winner-by-default = max scalar reward; user can click any Pareto-frontier point to swap in a different trade-off.

---

## 8. Data & privacy model

Two independent cloud pipelines:

| Pipeline | Trigger | Cloud storage | Retention |
|---|---|---|---|
| **Dashboard-sync** | Opt-in `dashboard_sync=True` | Every trace trickle-uploaded | Rolling last-N per tier (5/100/1000/∞) |
| **Optimization-burst** | Per optimization run | N traces for the run | **Purged on run completion** |

### Persists forever in cloud

- Optimization reports (tiered suggestions, summaries)
- Config diffs (what changed, what the baseline was)
- Aggregated scores (count, mean, p50/p95 of each evaluator)
- User feedback comments on suggestions
- Pareto-frontier scatter data (non-PII aggregate metrics)

### Does not persist

- Raw trace bodies used in an optimization run (purged on completion)
- User's LLM API keys (never transmitted at all)
- Trace bodies beyond the dashboard-sync rolling window

### Local (user's machine)

SQLite in the SDK is the **source of truth**. Every trace lands locally regardless of cloud sync. Users running `retune dashboard` against local SQLite always see 100% of their history, for free, forever.

---

## 9. Deliverable: the tiered apply-manifest

Every `OptimizationReport` has this structure:

```
▼ Optimizer's Understanding of Your Agent
  "I believe this is a customer-support RAG bot for billing
   questions. Goal: accurate, cited answers. [✏️ Correct this]"

▼ Summary
  Baseline judge score: 6.8 → Best candidate: 8.2  (+21%)
  Guardrails held: cost 1.1× · latency 0.94×

▼ Pareto Frontier [interactive 3D plot in dashboard]

▼ Tier 1 — One-Click Apply   [Apply All]  [Review Each]
  ☑ System prompt rewrite    impact: +0.8 judge, confidence: H
  ☑ Few-shot set (3 examples) impact: +0.4 judge, confidence: M
  ☑ retrieval_k: 5 → 8        impact: +0.3 judge, +5% latency

▼ Tier 2 — Copy-Paste Code Snippets
  [ ] Rewrite tool description for `lookup_invoice`
  [ ] Add query-rewriter preprocessing step

▼ Tier 3 — Conceptual Suggestions (manual engineering)
  • Split tool `manage_account` into 3 focused tools
  • Consider hybrid BM25+dense retrieval for FAQ corpus

▼ Your Feedback (feeds next optimization run)
  [text box]
```

### Tier definitions

- **Tier 1 (one-click apply)** — Things the SDK controls via `_config` at runtime: system prompt, few-shot examples, retrieval_k, reranker toggle, temperature, guardrail overrides. Applied by mutating wrapper config; takes effect on next `retuner.run()`.
- **Tier 2 (copy-paste code)** — Things the user exposes to the SDK but lives in their code: tool descriptions/schemas, retriever preprocessing, query rewriters. Report provides exact code snippet + target file hint.
- **Tier 3 (conceptual)** — Architectural guidance: agent graph changes, tool decomposition, retrieval strategy shifts, reindexing. Written recommendation + reasoning + expected impact + evidence. Never auto-applied.

### Each suggestion carries

- **Confidence:** H / M / L
- **Estimated impact:** Δ judge score, Δ cost, Δ latency
- **Axis:** prompt / tools / rag
- **Evidence:** link to the traces that motivated the suggestion

### Rejection behavior

Tier-1 rejections feed the Orchestrator's feedback memory for the next run. Rejection reason (free-text) is optional but encouraged. Reject-all still counts as 1 of the 15 free runs — compute was consumed — but the Orchestrator will prioritize different axes next run.

---

## 10. Quotas & billing enforcement

### Per-run limits by tier

| Limit | Free trial | Pro ($49) | Team ($99) | Enterprise |
|---|---|---|---|---|
| Runs total | **15 lifetime** | unlimited | unlimited | unlimited |
| Max candidates per run | 20 | 50 | 100 | custom |
| Max traces used per run | 50 | 200 | 1,000 | custom |
| Wall-clock cap per run | 15 min | 60 min | 180 min | custom |
| Cloud optimizer token budget per run | ~1M | ~5M | ~20M | custom |

### Enforcement

- Counter lives in **cloud DB**, not SDK memory. Server-side source of truth.
- Flow: SDK calls `/v1/optimize/preauthorize` → cloud reserves one slot → returns `run_id` → candidates execute → `/v1/optimize/commit` on completion deducts the slot. Infra failures auto-refund.
- `UsageGate` in the SDK is extended (not rewritten): gates at job-start, not per-candidate. Existing `/api/v1/billing/usage` endpoint extended with `{runs_used, runs_limit}`.
- No API key = no optimization. Closes the v0.2 "restart SDK to reset counter" loophole.

### Fair-use on "unlimited" plans

"Unlimited" in the table above means practically unlimited for normal usage, subject to a documented fair-use soft ceiling: ~100 runs/month on Pro, ~500 runs/month on Team (numbers settled in Phase 5). Exceeding the ceiling triggers a support conversation, not a hard block. Enterprise plans negotiate their own ceiling in the contract.

---

## 11. v1 implementation phasing

Build order, not scope cuts. All 5 subagents ship in v0.3.0.

| Phase | Weeks | Scope | Exit gate |
|---|---|---|---|
| **1. Infra** | 1–3 | Cloud optimizer service skeleton, long-poll SDK worker protocol, billing preauthorize/commit, JudgeAgent + declarative reward parser, ReportWriterAgent shell | Dogfood end-to-end with a noop optimizer |
| **2. PromptOptim** | 4–6 | Migrate Beam Search APO into PromptOptimizerAgent. Orchestrator v1 (prompt-only dispatch). | Internal alpha: prompt optimization works end-to-end |
| **3. ToolOptim** | 7–9 | ToolOptimizerAgent (introspects wrapped agent). Orchestrator learns to route. | Alpha extended: prompt + tools |
| **4. RAGOptim** | 10–12 | RAGOptimizerAgent (sweeps, reranker, indexing tier-3). | Private beta with 3–5 design-partner users |
| **5. Polish** | 13–16 | Pareto viz in dashboard, feedback-loop memory, tiered apply-manifest UI, dashboard rewrite, Stripe wiring, docs, public launch | **v0.3.0 GA** |

Each phase ends with a hard gate: at least one internal + one external user can execute that slice end-to-end. Miss the gate → do not advance.

### Dashboard rewrite note

The LangSmith-quality UI is built from scratch. Existing `retune-frontend/` is reference material only. Stack decision happens in Phase 1 and is documented in an addendum to this spec (React/Next remains the likely choice given existing team familiarity).

---

## 12. Migration from v0.2

### Breaking changes

- `Retuner(agent_purpose=..., api_key=...)` becomes **required** for `Mode.IMPROVE`. OBSERVE/EVALUATE modes unchanged.
- Evaluation is no longer gated by `UsageGate`. All evaluators (including `llm_judge`, `pairwise_judge`) run freely with user's LLM key. This matches the open-core repositioning. (Evaluation migration is covered in a separate spec.)
- Cloud endpoint `POST /api/v1/ingest/suggestions` is deprecated in favor of the new `/v1/optimize/*` family.

### Preserved / reused

- Existing `OptimizerDeepAgent` + Beam Search APO code is the core of `PromptOptimizerAgent`. Not rewritten.
- `UsageGate` class retained, semantics updated (per-run, not per-call).
- `CloudClient` retained for trace upload (dashboard-sync pipeline).
- `SQLiteStorage` retained as local source of truth.

### Schema migrations

- New table `optimization_runs(run_id, org_id, status, started_at, completed_at, report_url, slots_consumed)`.
- Existing `traces` table adds `optimization_run_id` (nullable) + TTL column for auto-purge of optimization-burst data.
- Add `organization_feedback(run_id, suggestion_id, accepted, user_comment)` for feedback-loop memory.

---

## 13. Risks & open questions for Phase 1

Resolved during brainstorm:

1. **Prompt-rewrite LLM:** user's choice (`rewriter_llm=` parameter, sensible default)
2. **Tool discovery mechanism:** introspection of wrapped agent at build time; never parse source files
3. **RAG & user code:** optimizer never modifies user code; uses runtime override hooks where the adapter supports them, else drops to Tier-2/3 suggestions
4. **Dashboard stack:** redo from scratch to LangSmith-quality; stack chosen in Phase 1

Still open, to be resolved early in Phase 1:

- Which LangChain/LangGraph retriever introspection hooks are stable across versions? (Affects which RAG changes can be Tier-1 vs Tier-2.)
- Stripe metered billing vs. flat-fee for Pro/Team — TBD in Phase 5.
- Telemetry for judge-rubric drift (how do we know the judge is still calibrated?) — design during Phase 1, implement Phase 5.

---

## 14. Success criteria for v0.3 launch

- A paying design-partner can take a previously-unoptimized LangChain RAG agent through a complete optimization run and accept ≥1 Tier-1 suggestion that measurably improves their judge score.
- 5 external users complete a free-trial optimization run in private beta without support intervention.
- Cloud optimizer cost per run stays under $2 of LLM spend on our side (not counting user's candidate-execution LLM spend, which is theirs).
- No raw trace data remains in cloud DB for users without `dashboard_sync=True`, 24 hours after any run completes.

---

## 15. Verification plan (end-to-end)

Described at the phase level in §11. At v0.3.0 GA:

1. Spin up a LangGraph RAG reference agent against a sample FAQ corpus.
2. Run it 50 times in `Mode.OBSERVE` to accumulate traces.
3. Trigger `retuner.optimize(source="last_n_traces", n=50, axes="auto")`.
4. Verify: Orchestrator dispatches all three subagents, candidates execute on SDK side, reward signal computes, report renders with all three tiers populated.
5. Verify: `dashboard_sync=False` case leaves cloud DB empty after purge.
6. Verify: 16th optimization run is rejected by `UsageGate` on the free trial.
7. Verify: dashboard Pareto plot renders; clicking a non-winner swaps recommendation.
8. Verify: feedback-box text persists into the next run's Orchestrator context.
