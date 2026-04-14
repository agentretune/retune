"""LangChain-backed RAG agent with intentionally oversized chunks.

Use in: 04_trial_rag.py — RAGOptimizerAgent should notice chunk_size=1500
is large and propose a chunk_sweep to 800.

In-memory vector store with a small synthetic corpus — no external deps
beyond langchain-openai (for embeddings fallback) or a dummy embedder.
"""
from __future__ import annotations

import os
from typing import Any


CORPUS = [
    "Refunds are available within 30 days of purchase. Contact billing@example.com with your invoice number. Partial refunds are prorated.",
    "To cancel your subscription, navigate to Account Settings → Billing → Manage Subscription → Cancel. Cancellation takes effect at the end of the current billing period.",
    "Payment methods can be updated via Account → Billing → Payment Methods. We accept Visa, Mastercard, American Express, and ACH.",
    "Annual billing provides a 20 percent discount versus monthly billing. Switching to annual is available at any time from the Billing page.",
    "Team members can be added via the Team page. Admins can invite new members by email; new members receive an invite link valid for 7 days.",
    "If your payment fails, you will receive an email notification. We retry the payment every 3 days for up to 2 weeks before downgrading the account.",
    "You can downgrade mid-cycle. The new plan takes effect at the next billing period; unused time is not refunded.",
    "Free trials last 14 days. No credit card is required to start a trial. Upgrade within the trial to keep your data.",
    "Invoices are available from the Billing → Invoice History page. They can be downloaded as PDF and are emailed automatically on each billing cycle.",
    "Billing emails can be changed from Account → Billing → Billing Email. This is independent of your login email.",
]


class _DummyEmbedder:
    """Trivial keyword-based 'embedding' — enough to make retrieval work."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        # 20-dim feature vector over common billing keywords
        keywords = [
            "refund", "cancel", "payment", "subscription", "billing",
            "invoice", "trial", "team", "discount", "annual",
            "fail", "retry", "plan", "downgrade", "upgrade",
            "admin", "email", "method", "period", "pdf",
        ]
        text_lower = text.lower()
        return [1.0 if k in text_lower else 0.0 for k in keywords]


class RAGAgent:
    """RAG agent with a retriever that optimizer can tune."""

    def __init__(self, k: int = 5, chunk_size: int = 1500) -> None:
        self.system_prompt = (
            "Answer the user's billing question using the retrieved context. "
            "Be concise. If the context doesn't cover the question, say so."
        )

        # Build in-memory vector store
        from langchain_core.vectorstores import InMemoryVectorStore
        vs = InMemoryVectorStore(_DummyEmbedder())
        vs.add_texts(CORPUS)

        self._retriever = vs.as_retriever(search_kwargs={"k": k})

        # Expose retrieval_config + retriever so introspect_retrieval_config can find them
        self.retrieval_config = {
            "retrieval_k": k,
            "chunk_size": chunk_size,
            "chunk_overlap": 200,
            "retrieval_strategy": "dense",
            "reranker_enabled": False,
        }
        self.retriever = self._retriever

        # LLM for the final answer
        self._llm = self._pick_llm()

    @staticmethod
    def _pick_llm() -> Any:
        if os.environ.get("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0.2)
        if os.environ.get("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        raise RuntimeError(
            "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY set — cannot build RAG agent."
        )

    def __call__(self, query: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        docs = self._retriever.invoke(query)
        context = "\n\n".join(f"- {d.page_content}" for d in docs)

        # Capture retrieval for ExecutionTrace.steps[]
        self._last_retrieval = {
            "type": "retrieval",
            "name": "vector_search",
            "output": [{"page_content": d.page_content} for d in docs],
            "duration_ms": 8.0,
        }

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        resp = self._llm.invoke(messages)
        return str(resp.content)


def make_rag_agent(k: int = 5, chunk_size: int = 1500) -> RAGAgent:
    return RAGAgent(k=k, chunk_size=chunk_size)


SAMPLE_QUERIES = [
    "How do I get a refund?",
    "Can I cancel my subscription?",
    "What payment methods do you accept?",
    "Is there an annual discount?",
    "How do I add team members?",
    "What happens if my payment fails?",
    "Can I downgrade my plan?",
    "Do you offer a free trial?",
    "Where are my invoices?",
    "How do I change my billing email?",
]
