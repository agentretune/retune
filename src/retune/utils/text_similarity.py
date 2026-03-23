"""Text similarity utilities -- n-gram overlap without external deps."""
from __future__ import annotations

_EMBEDDING_MODEL = None
_EMBEDDING_AVAILABLE: bool | None = None


def _check_embeddings() -> bool:
    """Check if sentence-transformers is available (cached)."""
    global _EMBEDDING_AVAILABLE
    if _EMBEDDING_AVAILABLE is None:
        try:
            import sentence_transformers  # noqa: F401
            _EMBEDDING_AVAILABLE = True
        except ImportError:
            _EMBEDDING_AVAILABLE = False
    return _EMBEDDING_AVAILABLE


def _get_embedding_model():
    """Lazy-load the embedding model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def embedding_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between sentence embeddings.

    Requires: pip install retune[embeddings]
    Raises ImportError if sentence-transformers not installed.
    """
    model = _get_embedding_model()
    embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(embeddings[0] @ embeddings[1])


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Best available semantic similarity. Uses embeddings if available, else n-grams."""
    if _check_embeddings():
        try:
            return embedding_similarity(text_a, text_b)
        except Exception:
            pass
    return text_overlap_score(text_a, text_b)


def _char_ngrams(text: str, n: int) -> set[str]:
    """Generate character n-grams from text."""
    return {text[i:i+n] for i in range(len(text) - n + 1)} if len(text) >= n else set()


def _word_ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    """Generate word n-grams."""
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)} if len(words) >= n else set()


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def text_overlap_score(source: str, target: str) -> float:
    """Compute text overlap score between source and target using n-grams.

    Uses a combination of character trigrams and word unigram/bigram overlap.
    Returns 0.0-1.0.
    """
    source = source.lower().strip()
    target = target.lower().strip()

    if not source or not target:
        return 0.0

    # Character n-gram similarity (trigrams)
    char_sim = _jaccard(_char_ngrams(source, 3), _char_ngrams(target, 3))

    # Word-level overlap
    src_words = source.split()
    tgt_words = target.split()

    # Unigram overlap
    unigram_sim = _jaccard(set(src_words), set(tgt_words))

    # Bigram overlap
    bigram_sim = _jaccard(_word_ngrams(src_words, 2), _word_ngrams(tgt_words, 2))

    # Weighted combination
    ngram_score = 0.3 * char_sim + 0.5 * unigram_sim + 0.2 * bigram_sim

    # Blend with embedding similarity if available
    if _check_embeddings():
        try:
            emb_sim = embedding_similarity(source, target)
            return 0.6 * emb_sim + 0.4 * ngram_score
        except Exception:
            pass

    return ngram_score


def text_is_referenced(source: str, target: str, threshold: float = 0.15) -> bool:
    """Check if source text is meaningfully referenced in target."""
    return text_overlap_score(source, target) > threshold


def information_contribution(source: str, target: str) -> float:
    """Directed recall: what fraction of target's content is covered by source."""
    if not source or not target:
        return 0.0
    source_words = set(source.lower().split())
    target_words = set(target.lower().split())
    if not target_words:
        return 0.0
    return len(source_words & target_words) / len(target_words)


def unique_information_score(
    source: str, target: str, other_sources: list[str]
) -> float:
    """Estimate how much info in target came ONLY from source.

    Computes word-level overlap between source and target that is NOT
    present in any other_sources. High score = unique causal contribution.
    """
    if not source or not target:
        return 0.0

    source_lower = source.lower()
    target_lower = target.lower()

    # Words from source that appear in target
    src_words = set(w for w in source_lower.split() if len(w) > 3)
    tgt_words = set(w for w in target_lower.split() if len(w) > 3)
    contribution = src_words & tgt_words

    if not contribution:
        return 0.0

    # Words from other sources that also appear in target
    other_coverage = set()
    for other in other_sources:
        other_words = set(w for w in other.lower().split() if len(w) > 3)
        other_coverage |= (other_words & tgt_words)

    # Unique contribution = what this source provides that no other does
    unique = contribution - other_coverage

    if not tgt_words:
        return 0.0

    return len(unique) / len(tgt_words)
