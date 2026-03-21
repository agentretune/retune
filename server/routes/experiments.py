"""Experiment API routes (stub for future implementation)."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def list_experiments():
    """List experiments (coming soon)."""
    return {"message": "Experiment management coming in v0.2", "experiments": []}


@router.post("/")
def create_experiment():
    """Create experiment (coming soon)."""
    return {"message": "Experiment management coming in v0.2"}
