# Retune end-to-end testing suite

Modular scripts for exercising every user-facing flow of Retune v0.3.0 against real infrastructure.

**This directory is separate from `tests/`** (unit tests). This folder is for **manual, interactive integration testing** — you run these scripts to see the product actually work.

---

## Layout

```
testing/
├── README.md                    ← you are here
├── setup/                       ← bootstrap local infra
│   ├── env.example              ← env vars you need
│   ├── 01_start_postgres.sh     ← local Postgres via Docker
│   ├── 02_init_db.sh            ← run Alembic migrations
│   ├── 03_bootstrap_org.py      ← create first org/user/API key
│   └── 04_start_cloud.sh        ← boot the cloud backend
├── agents/                      ← test agent fixtures
│   ├── echo_agent.py            ← simplest possible
│   ├── bad_prompt_agent.py      ← vague system prompt → PromptOptim has room
│   ├── tool_agent.py            ← one used tool + one unused → ToolOptim fires drop_tool
│   └── rag_agent.py             ← LangChain RAG with oversized chunks → RAGOptim suggests sweep
├── flows/                       ← scripted user journeys
│   ├── 01_pure_sdk_free.py      ← observe + evaluate locally, no cloud
│   ├── 02_trial_prompt.py       ← free trial: prompt axis only
│   ├── 03_trial_tool.py         ← free trial: tool axis only
│   ├── 04_trial_rag.py          ← free trial: RAG axis only
│   ├── 05_trial_combined.py     ← free trial: all 3 axes together
│   ├── 06_exhaust_trial.py      ← run 16 times, confirm 402 on 16th
│   ├── 07_feedback_loop.py      ← reject a suggestion, re-run, confirm passed to next
│   └── 08_apply_tier1.py        ← apply a Tier 1 suggestion, confirm state mutated
└── inspect/                     ← verification queries
    ├── db_state.py              ← print current DB state for the test org
    ├── cloud_health.py          ← smoke all cloud routes
    └── reset.py                 ← nuke the test org and start over
```

---

## Prerequisites (do these manually, once)

1. **Python 3.10+** and **Node 18+** installed.
2. **One LLM API key.** Recommend Anthropic (`claude-3-7-sonnet` is the default rewriter). OpenAI works too. Get from console.anthropic.com or platform.openai.com.
3. **Docker** (for local Postgres). Alternative: Supabase free tier.
4. **SDK installed**: from the repo root run `pip install -e ".[dev,langchain,langgraph,llm]"`.
5. **Cloud deps installed**: `cd retune-cloud && pip install -e ".[dev]"`.
6. **Frontend deps installed**: `cd retune-frontend && npm install`.

---

## Quick start (full walkthrough)

```bash
# 1. Copy env template and fill in your LLM API key
cp testing/setup/env.example testing/setup/.env
# Edit testing/setup/.env — set ANTHROPIC_API_KEY and/or OPENAI_API_KEY

# 2. Start infra (you do these manually because they need your system access)
bash testing/setup/01_start_postgres.sh      # starts Postgres in Docker
bash testing/setup/02_init_db.sh              # alembic upgrade head
python testing/setup/03_bootstrap_org.py      # creates test org + prints RETUNE_API_KEY
# Copy the printed RETUNE_API_KEY into testing/setup/.env

bash testing/setup/04_start_cloud.sh          # uvicorn on :8001 (leave this running)

# 3. In another terminal — start frontend
cd retune-frontend && npm run dev             # :5173 (leave running)

# 4. In a third terminal — run the flows
python testing/flows/01_pure_sdk_free.py      # no cloud needed
python testing/flows/02_trial_prompt.py       # uses the API key
python testing/flows/03_trial_tool.py
python testing/flows/04_trial_rag.py
python testing/flows/05_trial_combined.py

# 5. Open the dashboard
# http://localhost:5173 — log in with the email/password from step 2

# 6. Inspect
python testing/inspect/db_state.py            # pretty-print current DB
python testing/inspect/cloud_health.py        # check all routes return 2xx
```

---

## Map: "I want to test X" → "run this"

| I want to verify… | Run this |
|---|---|
| SDK works without any cloud | `flows/01_pure_sdk_free.py` |
| Prompt optimization end-to-end | `flows/02_trial_prompt.py` |
| Tool optimizer finds unused tools | `flows/03_trial_tool.py` |
| RAG optimizer suggests parameter sweeps | `flows/04_trial_rag.py` |
| All 3 axes dispatch in one run | `flows/05_trial_combined.py` |
| Free trial limit enforced | `flows/06_exhaust_trial.py` |
| Feedback feeds next run's context | `flows/07_feedback_loop.py` |
| Tier 1 apply mutates agent state | `flows/08_apply_tier1.py` |
| Dashboard UI renders correctly | `npm run dev` in `retune-frontend/` → open http://localhost:5173 |
| Stripe upgrade flow | `testing/stripe/` (add later) |

---

## What each flow prints

Every flow script:
1. States what scenario it's exercising at the top (scenario, expected outcome)
2. Runs the flow
3. Prints a **verification checklist** at the end: what to confirm manually or via `inspect/db_state.py`
4. Returns exit code 0 on success, 1 on obvious failure

This makes it easy to copy-paste a flow and modify it for your own testing.

---

## Known gaps

- **No Stripe flow scripted yet.** Use the Stripe CLI + dashboard manually (see the parent `docs/superpowers/plans/2026-04-13-optimizer-phase5-polish-dashboard-ga.md` §Task 11 for the manual runbook).
- **No LangGraph agent fixture.** The `rag_agent.py` uses LangChain; LangGraph-based agents would need their own fixture.
- **Cloud and frontend must be started manually** (Docker port conflicts, npm output, etc. are messy to script reliably across Windows/Mac/Linux). The `01_start_postgres.sh` and `04_start_cloud.sh` scripts are provided but YOU run them in terminals you control.
