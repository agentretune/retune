# Optimizer Phase 5 (Polish + Dashboard + Feedback Loop + GA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship v0.3.0 GA. Phase 5 is the polish phase: build the hosted dashboard UI from scratch (LangSmith-quality per the spec), wire Stripe billing end-to-end, add Pareto visualization, implement the human-feedback loop into the Orchestrator, clean up Dockerfile/deployment, and run the comprehensive manual smoke tests across all four axes.

**Architecture:** This phase is mostly UI + infrastructure, not new optimizer logic. Six conceptually distinct workstreams, each executable somewhat independently:

1. **Dashboard UI (LangSmith-quality)** — build a React/Next.js app in `retune-frontend/` from scratch. Views: optimization runs list, run detail, Pareto frontier plot (interactive), tiered apply-manifest viewer, feedback box.
2. **Cloud dashboard API** — read-only endpoints feeding the UI: `/v1/optimize/runs` (list), `/v1/optimize/{run_id}/pareto` (plot data), `/v1/optimize/{run_id}/feedback` (accept/reject + comment).
3. **Feedback loop memory** — Orchestrator reads prior-run feedback for the same org/agent and includes it in PromptOptimizerAgent's context at run start.
4. **Stripe billing wiring** — already scaffolded in `retune-cloud/server/routes/billing.py`. Phase 5 completes the checkout flow + webhook handling so upgrades/downgrades actually work and `optimize_runs_limit` updates on plan changes.
5. **Deployment cleanup** — `retune-cloud/pyproject.toml` missing prod deps (flagged in Phase 2 T12); `src/retune/__main__.py` references the deleted outer `server.app:app` and needs a pure-SDK local dashboard. Docker/Fly/Render configs need test.
6. **GA release** — version bump to 0.3.0, CHANGELOG, README updates, public announcement post.

**Tech Stack:** Cloud-side Python (FastAPI + existing stack). UI-side: React 18 + Vite + TypeScript + TailwindCSS + Recharts (Pareto). No new backend deps.

**Reference spec:** `docs/superpowers/specs/2026-04-12-optimizer-design.md` §9 (tiered apply-manifest presentation), §11 Phase 5 row.

**Baseline:** `phase2.1/multi-round` (207 tests, four subagents all shipped via routing).

---

## Scope

### In

- **Dashboard UI** — a fresh React/Vite app replacing `retune-frontend/`. Views: Dashboard home (recent runs), Run detail (tiered suggestions + Pareto plot + markdown report + feedback box), Login/API key management, Billing/Upgrade page.
- **Cloud dashboard API** — 4 new routes under `/v1/optimize/`:
  - `GET /runs` (paginated org history)
  - `GET /{run_id}/pareto` (plot data)
  - `POST /{run_id}/feedback` (free-text + per-suggestion accept/reject)
  - `GET /{run_id}/feedback` (retrieve)
- **Feedback loop memory** — `db.get_recent_feedback(org_id, limit)` helper; Orchestrator includes these in PromptOptimizerAgent context when present.
- **Stripe completion** — `/v1/billing/webhook` handling: `checkout.session.completed` → `update_org_plan` (lifts `optimize_runs_limit` from 15 to 100/500/unlimited per plan); `customer.subscription.deleted` → downgrade to free + resets `optimize_runs_used` to 0 for the new billing month.
- **Deployment polish** — `retune-cloud/pyproject.toml` gets prod deps (psycopg2-binary, pyjwt, bcrypt, stripe, python-dotenv). `src/retune/__main__.py dashboard` uses a new pure-SDK local FastAPI app (no cloud dependency).
- **GA** — SDK version bump to 0.3.0, CHANGELOG, README refresh.

### Out (deferred)

- SSO/SAML enterprise auth (already scaffolded in `retune-cloud/server/routes/sso.py`, keep as-is)
- A/B testing of applied suggestions in prod traffic (v0.4)
- Cloud-hosted LLM-judge evaluator (still BYO-key, matches the open-core positioning)
- Mobile app

---

## File Structure

### New — Cloud backend

| File | Responsibility |
|---|---|
| `retune-cloud/server/routes/optimize_dashboard.py` | `GET /v1/optimize/runs`, `GET /v1/optimize/{run_id}/pareto`, `POST /v1/optimize/{run_id}/feedback`, `GET /v1/optimize/{run_id}/feedback` |
| `retune-cloud/server/optimizer/feedback.py` | `FeedbackEntry` Pydantic model + `record_feedback()`, `list_recent_feedback(org_id, limit)` helpers |

### Modified — Cloud backend

| File | Change |
|---|---|
| `retune-cloud/server/db/schema.sql` | Add `optimization_feedback` table (persistent; survives run-trace purge) |
| `retune-cloud/server/db/postgres.py` | Feedback helpers: `save_feedback`, `get_run_feedback`, `get_recent_org_feedback` |
| `retune-cloud/server/app.py` | Register new `optimize_dashboard` router |
| `retune-cloud/server/routes/billing.py` | Complete Stripe webhook handler — update `optimize_runs_limit` on plan change |
| `retune-cloud/server/optimizer/orchestrator.py` | Load recent org feedback and include in PromptOptimizerAgent context |
| `retune-cloud/pyproject.toml` | Add `psycopg2-binary`, `pyjwt`, `bcrypt`, `stripe`, `python-dotenv` to `[project] dependencies` |

### New — SDK (outer repo)

| File | Responsibility |
|---|---|
| `src/retune/dashboard/__init__.py` | Local pure-SDK FastAPI app for `retune dashboard` CLI |
| `src/retune/dashboard/app.py` | FastAPI app reading from local SQLite storage |
| `src/retune/dashboard/templates/` | Minimal HTML templates (traces list, trace detail) |

### Modified — SDK

| File | Change |
|---|---|
| `src/retune/__main__.py` | `dashboard` subcommand points at `retune.dashboard.app:app` instead of deleted `server.app:app` |
| `pyproject.toml` | Version → 0.3.0; add `jinja2>=3.0` to `[project.optional-dependencies].server` |
| `CHANGELOG.md` | Phase 1–5 summary |
| `README.md` | Updated Installation, Quickstart with new optimize API, CLI section |

### New — Frontend (private)

Full rewrite of `retune-frontend/`:

| Path | Responsibility |
|---|---|
| `retune-frontend/package.json` | Vite + React 18 + TypeScript + Tailwind + Recharts + SWR |
| `retune-frontend/src/App.tsx` | Router: `/login`, `/runs`, `/runs/:run_id`, `/billing` |
| `retune-frontend/src/pages/RunsList.tsx` | Paginated table of optimization runs |
| `retune-frontend/src/pages/RunDetail.tsx` | Tabbed view: Overview / Suggestions / Pareto / Feedback |
| `retune-frontend/src/pages/Billing.tsx` | Current plan, usage bars, upgrade button (Stripe Checkout) |
| `retune-frontend/src/components/ParetoScatter.tsx` | 3D-ish scatter via Recharts (quality × cost × latency, Pareto frontier highlighted) |
| `retune-frontend/src/components/TieredManifest.tsx` | Collapsible Tier 1 / Tier 2 / Tier 3 lists with accept/reject/copy buttons |
| `retune-frontend/src/lib/api.ts` | API client hitting `https://api.agentretune.com/v1/optimize/...` |

### Tests

| File | Covers |
|---|---|
| `retune-cloud/tests/test_optimize_dashboard_routes.py` | All 4 new dashboard routes (mocked DB) |
| `retune-cloud/tests/test_feedback_db.py` | Feedback helpers |
| `retune-cloud/tests/test_stripe_webhook.py` | Webhook → plan update |
| `retune-cloud/tests/test_orchestrator_feedback_memory.py` | Orchestrator passes recent feedback to PromptOptim |
| `tests/unit/test_sdk_dashboard_local.py` | Local SDK dashboard app serves trace list without cloud deps |
| `retune-frontend/src/__tests__/RunDetail.test.tsx` | UI unit tests (Vitest) — render + interaction |
| `tests/integration/test_v030_smoke.py` | Comprehensive end-to-end smoke test across all 4 axes |

---

## Task Summary

Fifteen tasks across six workstreams. Tasks 1–5 are cloud-backend (can execute in sequence). Tasks 6–10 are frontend (can execute in parallel if multi-session). Tasks 11–15 are polish + release.

### Workstream A — Feedback loop (Tasks 1–3)

1. Cloud DB: `optimization_feedback` table + helpers
2. Cloud route: `POST/GET /v1/optimize/{run_id}/feedback` + Pydantic model
3. Orchestrator reads recent org feedback, includes in PromptOptim context

### Workstream B — Dashboard backend API (Tasks 4–5)

4. `GET /v1/optimize/runs` (paginated org run history)
5. `GET /v1/optimize/{run_id}/pareto` (shaped for Recharts)

### Workstream C — Frontend rewrite (Tasks 6–10)

6. Scaffold Vite + React + TS + Tailwind + Recharts in `retune-frontend/`
7. Login + API key management page
8. Runs list view
9. Run detail view (tabbed: Overview / Suggestions / Pareto / Feedback)
10. Billing page + Stripe Checkout button

### Workstream D — Stripe completion (Task 11)

11. Complete `POST /v1/billing/webhook` to update `optimize_runs_limit` on plan change

### Workstream E — Deployment polish (Tasks 12–13)

12. `retune-cloud/pyproject.toml` adds prod deps; `Dockerfile` verified; Fly/Render smoke
13. SDK `retune dashboard` points at new `retune.dashboard.app:app` (pure-SDK local FastAPI)

### Workstream F — GA release (Tasks 14–15)

14. Version bump to 0.3.0 + CHANGELOG + README refresh
15. v0.3.0 smoke test across all axes + PyPI publish

---

## Task 1: Feedback DB + helpers

**Files:**
- Modify: `retune-cloud/server/db/schema.sql`
- Modify: `retune-cloud/server/db/postgres.py`
- Test: `retune-cloud/tests/test_feedback_db.py`

- [ ] **Step 1.1: Schema**

```sql
-- ============ Phase 5: Feedback memory (persistent) ============

CREATE TABLE IF NOT EXISTS optimization_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id VARCHAR(64) REFERENCES optimization_runs(id) ON DELETE SET NULL,
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    suggestion_id VARCHAR(128),
    tier INTEGER CHECK (tier IN (1, 2, 3)),
    accepted BOOLEAN,
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_org ON optimization_feedback(org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_run ON optimization_feedback(run_id);
```

- [ ] **Step 1.2: Helpers** — pattern matches existing Phase 1–4 helpers:

```python
def save_feedback(
    run_id: str | None, org_id: str, suggestion_id: str | None,
    tier: int | None, accepted: bool | None, comment: str | None,
) -> None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO optimization_feedback
                    (run_id, org_id, suggestion_id, tier, accepted, comment)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (run_id, org_id, suggestion_id, tier, accepted, comment),
            )
        conn.commit()
    finally:
        put_conn(conn)


def get_run_feedback(run_id: str) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT suggestion_id, tier, accepted, comment, created_at "
                "FROM optimization_feedback WHERE run_id = %s ORDER BY created_at DESC",
                (run_id,),
            )
            return [
                {"suggestion_id": r[0], "tier": r[1], "accepted": r[2],
                 "comment": r[3], "created_at": str(r[4]) if r[4] else None}
                for r in cur.fetchall()
            ]
    finally:
        put_conn(conn)


def get_recent_org_feedback(org_id: str, limit: int = 20) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT run_id, suggestion_id, tier, accepted, comment, created_at "
                "FROM optimization_feedback WHERE org_id = %s "
                "ORDER BY created_at DESC LIMIT %s",
                (org_id, limit),
            )
            return [
                {"run_id": r[0], "suggestion_id": r[1], "tier": r[2],
                 "accepted": r[3], "comment": r[4],
                 "created_at": str(r[5]) if r[5] else None}
                for r in cur.fetchall()
            ]
    finally:
        put_conn(conn)
```

- [ ] **Step 1.3: 3 tests** following Phase 1 T3 pattern (mocked connection, save asserts INSERT SQL, get variants).

- [ ] **Step 1.4: Commit**

---

## Task 2: Feedback route

**Files:**
- Create: `retune-cloud/server/routes/feedback.py`
- Modify: `retune-cloud/server/app.py` (include router)
- Test: `retune-cloud/tests/test_feedback_routes.py`

Endpoints:
- `POST /v1/optimize/{run_id}/feedback` body `{suggestion_id, tier, accepted, comment}` → stores
- `GET /v1/optimize/{run_id}/feedback` → list

Validation: org must own run. Standard `require_auth` + `_verify_run_belongs_to_org`.

2 tests (happy path + 403 on wrong org).

---

## Task 3: Orchestrator uses feedback memory

**Files:**
- Modify: `retune-cloud/server/optimizer/orchestrator.py`
- Modify: `retune-cloud/server/optimizer/prompt_optimizer/agent.py` (accept `prior_feedback` param)
- Test: `retune-cloud/tests/test_orchestrator_feedback_memory.py`

- [ ] Orchestrator reads `db.get_recent_org_feedback(row["org_id"], limit=10)` at run start
- [ ] Passes the feedback list to `PromptOptimizerAgent(rewriter_llm=..., prior_feedback=[...])`
- [ ] Agent's constructor stores feedback; `run_iterative` prepends a formatted summary into the critique prompts (e.g. "Prior feedback from the user: ...")
- [ ] 1 new test asserts feedback is loaded and passed to PromptOptim

---

## Task 4: Runs list endpoint

**Files:**
- Create: `retune-cloud/server/routes/optimize_dashboard.py`
- Modify: `retune-cloud/server/app.py`
- Test: `retune-cloud/tests/test_optimize_dashboard_runs_list.py`

```python
@router.get("/runs")
def list_runs(
    authorization: str | None = Header(None),
    limit: int = 50,
    offset: int = 0,
) -> dict:
    payload = require_auth(authorization)
    rows = db.list_opt_runs(payload["org"], limit=limit, offset=offset)
    return {"runs": rows, "total": db.count_opt_runs(payload["org"])}
```

Add `db.list_opt_runs(org_id, limit, offset)` and `db.count_opt_runs(org_id)` helpers too.

1 test.

---

## Task 5: Pareto endpoint

**Files:**
- Modify: `retune-cloud/server/routes/optimize_dashboard.py`
- Test: `retune-cloud/tests/test_optimize_dashboard_pareto.py`

`GET /v1/optimize/{run_id}/pareto` returns:

```json
{
  "points": [
    {"candidate_id": "cb", "quality": 5.0, "cost": 0.001, "latency": 1.0, "is_pareto": true},
    ...
  ],
  "baseline_candidate_id": "cb"
}
```

Where `is_pareto` is True iff no other point strictly dominates this one in (quality desc, cost asc, latency asc). Frontend colors Pareto-frontier points in a distinct hue.

1 test with hand-crafted pareto data + expected `is_pareto` flags.

---

## Task 6: Scaffold frontend

**Files:**
- Delete: existing `retune-frontend/` (stash the old for reference if needed)
- Create: new Vite project

```bash
cd retune-frontend
npm create vite@latest . -- --template react-ts
npm install -D tailwindcss postcss autoprefixer vitest @testing-library/react
npm install react-router-dom swr recharts
npx tailwindcss init -p
```

Configure Tailwind, basic layout, routing scaffold. Commit.

(This is a scaffolding task with no tests — visual inspection only.)

---

## Task 7: Login + API key page

**Files:**
- Create: `retune-frontend/src/pages/Login.tsx`
- Create: `retune-frontend/src/lib/auth.ts` (local storage token + /v1/auth/login call)

Minimal login form → POST `/v1/auth/login` → stash JWT. One Vitest test.

---

## Task 8: Runs list view

**Files:**
- Create: `retune-frontend/src/pages/RunsList.tsx` (SWR fetch of `/v1/optimize/runs`)
- Create: `retune-frontend/src/components/RunRow.tsx` (status badge, improvement %, timestamp)

Renders paginated table. One Vitest test on the cell rendering.

---

## Task 9: Run detail view

**Files:**
- Create: `retune-frontend/src/pages/RunDetail.tsx` (tabbed layout)
- Create: `retune-frontend/src/components/TieredManifest.tsx` (accept/reject/copy buttons)
- Create: `retune-frontend/src/components/ParetoScatter.tsx` (Recharts scatter with Pareto highlight)
- Create: `retune-frontend/src/components/FeedbackBox.tsx`

POSTs feedback on accept/reject click, textarea for free-form comments.

Three Vitest tests (tier rendering, Pareto hover, feedback submission).

---

## Task 10: Billing page + Stripe Checkout

**Files:**
- Create: `retune-frontend/src/pages/Billing.tsx`
- Backend: `GET /v1/billing/usage` (already exists), `POST /v1/billing/checkout` (already exists)

Renders `runs_used/runs_limit` as a progress bar; upgrade button hits `/v1/billing/checkout` and redirects to Stripe-hosted checkout.

One Vitest test.

---

## Task 11: Stripe webhook completes plan updates

**Files:**
- Modify: `retune-cloud/server/routes/billing.py` — the `webhook` handler
- Test: `retune-cloud/tests/test_stripe_webhook.py`

Currently (per Phase 1 baseline inspection) the webhook handles `checkout.session.completed` but may not touch `optimize_runs_limit`. Add:

```python
# Inside the checkout.session.completed handler, after update_org_plan(...):
    new_limit = {
        "pro": 100,
        "team": 500,
        "enterprise": 10_000,
    }.get(plan, 15)
    db.update_org_optimize_runs_limit(org_id, new_limit)
    db.reset_org_optimize_runs_used(org_id)
```

Add those two DB helpers; 2 tests (webhook fires update, bad signature rejected).

---

## Task 12: Cloud deployment polish

**Files:**
- Modify: `retune-cloud/pyproject.toml` (add prod deps)
- Verify: `Dockerfile` builds clean

Add to dependencies:
```
psycopg2-binary>=2.9
pyjwt>=2.8
bcrypt>=4.0
stripe>=7.0
python-dotenv>=1.0
```

Local verification: `docker build -t retune-cloud .` should succeed without imperative `pip install` commands in the Dockerfile falling out of sync.

Also update Dockerfile to install from the pyproject.toml rather than imperative `pip install retune[...]`.

No new tests — just verify the import chain works.

---

## Task 13: SDK local dashboard (pure-SDK FastAPI)

**Files:**
- Create: `src/retune/dashboard/__init__.py`
- Create: `src/retune/dashboard/app.py` (minimal FastAPI serving `SQLiteStorage` traces)
- Create: `src/retune/dashboard/templates/` (jinja2 templates — just a traces list)
- Modify: `src/retune/__main__.py` — point `dashboard` command at `retune.dashboard.app:app`
- Test: `tests/unit/test_sdk_dashboard_local.py`

Minimal FastAPI app:

```python
# src/retune/dashboard/app.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from retune.storage import SQLiteStorage

app = FastAPI(title="Retune Local Dashboard")


@app.get("/", response_class=HTMLResponse)
def home():
    # Storage path from env; default ./retune.db
    import os
    storage = SQLiteStorage(path=os.environ.get("RETUNE_STORAGE_PATH", "./retune.db"))
    traces = storage.get_traces(limit=100)
    rows = "\n".join(
        f"<tr><td>{t.get('query', '')[:40]}</td><td>{t.get('response', '')[:40]}</td></tr>"
        for t in traces
    )
    return f"""
    <html><head><title>Retune Dashboard</title></head>
    <body>
      <h1>Recent traces</h1>
      <table border=1>
        <tr><th>Query</th><th>Response</th></tr>
        {rows}
      </table>
    </body></html>
    """
```

Two tests: dashboard returns HTML on root, handles empty storage.

---

## Task 14: Version bump + changelog + README

**Files:**
- Modify: `pyproject.toml` — `version = "0.3.0"`
- Modify: `CHANGELOG.md`
- Modify: `README.md`

**CHANGELOG.md** additions:

```md
## v0.3.0 - 2026-XX-XX

### Major changes — open-core repositioning
- Observability and all evaluators are now fully in the open-source SDK.
  BYO-LLM-key for llm_judge / pairwise_judge.
- Cloud-hosted Optimization is now the paid pillar, with 15 free trial runs
  for any user with a cloud API key.

### Optimizer
- New `Retuner.optimize(source, n, axes, ...)` method orchestrates a
  cloud-side optimization run across three axes:
  - `prompt`: Beam Search APO with LLM-proposed rewrites + few-shot curation
  - `tools`: drop unused + rewrite descriptions + tighten schemas
  - `rag`:   parameter sweep over k, chunk_size, reranker, strategy
- Multi-round beam search with in-loop scoring via JudgeAgent
- Tiered apply-manifest (Tier 1 auto-apply, Tier 2 copy-paste, Tier 3 conceptual)
- Human feedback loop: accepted/rejected suggestions feed the next run

### Infrastructure
- Hosted dashboard UI (LangSmith-quality), Pareto frontier visualization
- Stripe billing wired end-to-end for Free → Pro → Team → Enterprise plans

### Breaking changes
- `agent_purpose=` is required when `mode=Mode.IMPROVE`
- `retune dashboard` CLI now serves a pure-SDK local app; the previous
  cloud-backed dashboard moved to `agentretune.com`
```

**README.md** updates: replace old OptimizerDeepAgent examples with the new `Retuner.optimize(...)` API. Add a section on the three axes. Update the feature table: observability ✓ free, evaluation ✓ free, optimization ⚡ 15 free then paid.

---

## Task 15: v0.3.0 smoke test + release

**Files:**
- Create: `tests/integration/test_v030_smoke.py` — comprehensive integration across all four axes

```python
"""v0.3.0 comprehensive smoke test.

Exercises: observe → evaluate → optimize across all three axes → feedback."""
```

The smoke test uses fake_db + mocked subagents (like prior E2E tests), asserting:
1. Preauthorize accepts traces + tool_metadata + retrieval_config
2. Orchestrator dispatches all three subagents
3. Report has suggestions across all axes
4. Feedback submission stored, retrieved on next run
5. Manual assertion that `runs_used` incremented, slot committed

Then:
- [ ] Tag `v0.3.0-rc1` on outer repo main after Phase 5 branch merges
- [ ] PyPI publish (after manual testing against `https://api.agentretune.com` staging)
- [ ] GitHub release notes = CHANGELOG excerpt
- [ ] Public announcement (blog / social)

---

## Phase 5 exit gate

- [ ] All tests pass (Phase 2.1's 207 + ~30 new = ~237)
- [ ] `ruff check` clean
- [ ] `mypy` no new errors
- [ ] Docker build succeeds on both Linux and the dev machine
- [ ] Fly or Render deployment smoke test: real HTTP to `https://api-staging.agentretune.com/v1/optimize/runs` returns 200
- [ ] Manual E2E: real LangChain RAG agent → 15 OBSERVE runs → `retuner.optimize(axes=["prompt","tools","rag"])` → report in dashboard shows all three tiers populated, apply Tier 1 works
- [ ] Free trial exhaustion: 16th run returns 402
- [ ] Stripe: upgrade to Pro → 17th run succeeds (limit bumped to 100)
- [ ] Feedback flow: reject a suggestion, run again → rejection reason appears in Orchestrator context passed to PromptOptim

Phase 5 complete → **v0.3.0 GA** ships.

---

## Known risks

1. **Frontend rewrite is the biggest task.** Scaffolding a LangSmith-quality UI from scratch is 3–4 weeks of focused work even with experienced React developers. This plan gives it 5 tasks (6–10) which is a realistic slice but not exhaustive — polish iterations beyond Phase 5 are expected.
2. **Stripe webhook testing in CI.** The `test_stripe_webhook.py` tests fake the webhook signature. Real Stripe sends actual signed events; manual testing against a Stripe test-mode account is required before GA.
3. **Docker-from-pyproject.toml** introduces a difference vs. the prior imperative install. The Render/Fly deployments should be smoke-tested before merging to main.
4. **Migration from v0.2 users.** Anyone on v0.2 calling the old `OptimizerDeepAgent` API will need to migrate. The CHANGELOG lists the breaking changes; README should link to a migration guide (or this plan's §12 can be promoted).
5. **Cloud-hosted LLM judge is deferred.** Pro users currently BYO-key for LLM judge. Some users expect a hosted judge as part of "paid" — document this trade-off in the Pro plan description.
