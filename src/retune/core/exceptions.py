"""Custom exceptions for Retune."""

from __future__ import annotations


class RetuneError(Exception):
    """Base exception for Retune."""


class AdapterError(RetuneError):
    """Raised when an adapter encounters an error."""


class AdapterNotFoundError(RetuneError):
    """Raised when a requested adapter is not registered."""


class EvaluatorError(RetuneError):
    """Raised when an evaluator encounters an error."""


class OptimizerError(RetuneError):
    """Raised when the optimizer encounters an error."""


class StorageError(RetuneError):
    """Raised when storage operations fail."""


class ConfigError(RetuneError):
    """Raised for configuration errors."""


class AgentError(RetuneError):
    """Raised when a deep agent encounters an error."""
