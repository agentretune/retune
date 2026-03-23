"""Retune Dashboard — FastAPI backend."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware

from server.config import server_settings
from server.routes import (
    ab_experiments,
    alerts,
    audit,
    auth_routes,
    auto_optimization,
    billing,
    configs,
    datasets,
    evals,
    hosted_eval,
    ingest,
    prompt_versions,
    sessions,
    sso,
    traces,
)

app = FastAPI(
    title="Retune Dashboard API",
    description="Self-improving agent optimization platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=server_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(traces.router, prefix="/api/v1/traces", tags=["traces"])
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["sessions"])
app.include_router(configs.router, prefix="/api/v1/configs", tags=["configs"])
app.include_router(evals.router, prefix="/api/v1/evals", tags=["evaluations"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
app.include_router(auth_routes.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(billing.router, prefix="/api/v1/billing", tags=["billing"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])
app.include_router(hosted_eval.router, prefix="/api/v1/hosted", tags=["hosted"])
app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["datasets"])
app.include_router(prompt_versions.router, prefix="/api/v1/prompts", tags=["prompts"])
app.include_router(ab_experiments.router, prefix="/api/v1/experiments", tags=["experiments"])
app.include_router(auto_optimization.router, prefix="/api/v1/auto-opt", tags=["auto-optimization"])
app.include_router(sso.router, prefix="/api/v1/sso", tags=["sso"])
app.include_router(audit.router, prefix="/api/v1/audit", tags=["audit"])


@app.on_event("startup")
def startup_event():
    try:
        from server.scheduler import start_scheduler
        start_scheduler()
    except Exception:
        pass  # Scheduler optional


# Auth verification (used by SDK cloud client)
@app.get("/api/v1/auth/verify")
def verify_auth(authorization: str | None = Header(None)):
    from server.routes.ingest import _verify_api_key
    _verify_api_key(authorization)
    return {"status": "ok"}


@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}


# Serve the React dashboard
dashboard_dir = Path(__file__).parent.parent / "dashboard" / "dist"
if dashboard_dir.exists():
    from fastapi.responses import HTMLResponse

    _index_html = (dashboard_dir / "index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    def serve_dashboard():
        return HTMLResponse(content=_index_html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=server_settings.host, port=server_settings.port)
