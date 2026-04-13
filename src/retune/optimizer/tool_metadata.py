"""Serializable tool-metadata envelope for SDK→cloud upload."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolMetadata(BaseModel):
    """One wrapped agent's tool, in serializable form."""
    name: str
    description: str = ""
    args_schema: dict[str, Any] = Field(default_factory=dict)
    is_async: bool = False
