"""Example: Wrapping a LangChain RAG pipeline with Retune.

Requires:
    pip install retune[langchain,llm]
    pip install langchain-community faiss-cpu
"""

from dotenv import load_dotenv

load_dotenv()

from retune import Retuner, Mode, OptimizationConfig


def main():
    # --- Build a LangChain RAG pipeline ---
    from langchain_community.vectorstores import FAISS
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    # Create a simple vector store
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "RAG combines retrieval with generation for better answers.",
        "LangChain is a framework for building LLM applications.",
        "Agents can use tools to interact with external systems.",
    ]
    vectorstore = FAISS.from_texts(texts, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Build the RAG chain
    prompt = ChatPromptTemplate.from_template(
        "Answer based on the context:\n\n{context}\n\nQuestion: {question}"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- Wrap with Retune ---
    wrapped = Retuner(
        agent=rag_chain,
        adapter="langchain",
        mode=Mode.EVALUATE,
        evaluators=["llm_judge", "retrieval", "latency"],
        config=OptimizationConfig(top_k=3, temperature=0.3),
    )

    # --- Run queries ---
    queries = [
        "What is machine learning?",
        "How does RAG work?",
        "What are agents?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        response = wrapped.run(query)
        print(f"Answer: {response.output[:200]}")
        for r in response.eval_results:
            print(f"  [{r.evaluator_name}] Score: {r.score:.2f} — {r.reasoning}")

    # --- View summary ---
    print(f"\n{'='*60}")
    summary = wrapped.get_eval_summary()
    print(f"Total traces: {summary['total_traces']}")
    for name, stats in summary["scores"].items():
        print(f"  {name}: mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

    # --- Try improve mode ---
    wrapped.set_mode(Mode.IMPROVE)
    response = wrapped.run("Explain deep learning")
    if response.suggestions:
        print(f"\nSuggestions:")
        for s in response.suggestions:
            print(f"  {s.param_name}: {s.old_value} → {s.new_value}")
            print(f"    Reason: {s.reasoning}")


if __name__ == "__main__":
    main()
