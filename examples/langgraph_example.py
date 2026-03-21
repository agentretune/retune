"""Example: Wrapping a LangGraph agent with Retune.

Requires:
    pip install retune[langgraph,llm]
    pip install langchain-community
"""

from dotenv import load_dotenv

load_dotenv()

from retune import Retuner, Mode, OptimizationConfig


def main():
    from typing import Annotated, TypedDict

    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages

    # --- Define state ---
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    # --- Build a simple LangGraph agent ---
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    def chatbot(state: AgentState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("chatbot", chatbot)
    graph.set_entry_point("chatbot")
    graph.add_edge("chatbot", END)

    compiled = graph.compile()

    # --- Wrap with Retune ---
    wrapped = Retuner(
        agent=compiled,
        adapter="langgraph",
        mode=Mode.OBSERVE,
        config=OptimizationConfig(temperature=0.3),
    )

    # --- Run ---
    response = wrapped.run("What is the capital of France?")
    print(f"Output: {response.output}")

    # Check trace
    traces = wrapped.get_traces()
    if traces:
        trace = traces[0]
        print(f"\nTrace ID: {trace.trace_id}")
        print(f"Steps: {len(trace.steps)}")
        for step in trace.steps:
            print(f"  [{step.step_type.value}] {step.name} — {step.duration_ms:.0f}ms")
        print(f"Total duration: {trace.duration_ms:.0f}ms")


if __name__ == "__main__":
    main()
