"""Quickstart — wrap any callable with Retune in 5 lines."""

from retune import Retuner, Mode


# 1. Define any agent/pipeline as a simple function
def my_agent(query: str) -> str:
    """Simulated agent — replace with your real agent."""
    return f"The answer to '{query}' is: This is a simulated response."


# 2. Wrap it
wrapped = Retuner(
    agent=my_agent,
    adapter="custom",  # Use "langchain" or "langgraph" for real frameworks
    mode=Mode.OBSERVE,  # Start with observe, upgrade when ready
)

# 3. Run it — same as before, but now you get traces
response = wrapped.run("What is machine learning?")
print(f"Output: {response.output}")
print(f"Mode: {response.mode}")

# 4. Check traces
traces = wrapped.get_traces()
print(f"\nTraces captured: {len(traces)}")
if traces:
    trace = traces[0]
    print(f"  Query: {trace.query}")
    print(f"  Steps: {len(trace.steps)}")
    print(f"  Duration: {trace.duration_ms:.0f}ms")

# 5. Upgrade to evaluate mode when ready
wrapped.set_mode(Mode.EVALUATE)
# Now add evaluators (requires pip install retune[llm]):
# wrapped = Retuner(
#     agent=my_agent,
#     adapter="custom",
#     mode=Mode.EVALUATE,
#     evaluators=["llm_judge"],
# )
