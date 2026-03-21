"""Built-in tools for Retune deep agents."""

from retune.tools.base import RetuneTool


def get_builtin_tools() -> list[RetuneTool]:
    """Return all built-in tools."""
    from retune.tools.builtin.config_search import ConfigSearchTool
    from retune.tools.builtin.credit_assigner import CreditAssignerTool
    from retune.tools.builtin.gradient_aggregator import GradientAggregatorTool
    from retune.tools.builtin.metrics import MetricsCalculatorTool
    from retune.tools.builtin.prompt_analyzer import PromptAnalyzerTool
    from retune.tools.builtin.prompt_rewriter import PromptRewriterTool
    from retune.tools.builtin.rollout_runner import RolloutRunnerTool
    from retune.tools.builtin.trace_reader import TraceReaderTool

    return [
        TraceReaderTool(),
        MetricsCalculatorTool(),
        PromptAnalyzerTool(),
        CreditAssignerTool(),
        ConfigSearchTool(),
        PromptRewriterTool(),
        RolloutRunnerTool(),
        GradientAggregatorTool(),
    ]
