# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
