"""FastAPI application — SaaS dashboard backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import server_settings
from server.routes import experiments, traces

app = FastAPI(
    title="AutoMLAgent Dashboard API",
    description="Self-Improving Agent & RAG Optimization Platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=server_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(traces.router, prefix="/api/v1/traces", tags=["traces"])
app.include_router(experiments.router, prefix="/api/v1/experiments", tags=["experiments"])


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=server_settings.host, port=server_settings.port)
