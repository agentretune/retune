"""Shared test fixtures."""

import pytest

from retune.core.enums import Mode
from retune.core.models import OptimizationConfig
from retune.storage.sqlite_storage import SQLiteStorage


def mock_agent(query: str) -> str:
    """Simple mock agent for testing."""
    return f"Mock response to: {query}"


@pytest.fixture
def sample_agent():
    return mock_agent


@pytest.fixture
def tmp_storage(tmp_path):
    db_path = str(tmp_path / "test.db")
    return SQLiteStorage(db_path)


@pytest.fixture
def sample_config():
    return OptimizationConfig(
        top_k=4,
        chunk_size=500,
        temperature=0.5,
    )
