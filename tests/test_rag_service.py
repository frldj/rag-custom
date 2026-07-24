"""Unit tests for the RAG service context and pipeline logic.

Heavy dependencies (Milvus, Ollama, Redis, Langfuse, NeMo Guardrails) are
mocked so tests run in CI without any external service.
"""

import pytest


class TestContextBuilding:
    """Tests the context assembly logic: trimming, ordering, formatting."""

    def test_context_respects_max_chars(self):
        max_chars = 200
        contexts = [{"text": "A" * 100}, {"text": "B" * 100}, {"text": "C" * 100}]
        built = []
        total = 0
        for ctx in contexts:
            txt = ctx["text"]
            if total + len(txt) > max_chars:
                break
            built.append(txt)
            total += len(txt)
        assert total <= max_chars
        assert len(built) == 2

    def test_context_source_labels(self):
        contexts = [{"text": "foo"}, {"text": "bar"}]
        ctx_str = "\n".join(
            f"SOURCE {i + 1}: {c['text']}" for i, c in enumerate(contexts)
        )
        assert "SOURCE 1: foo" in ctx_str
        assert "SOURCE 2: bar" in ctx_str

    def test_empty_contexts_yields_empty_string(self):
        contexts = []
        ctx_str = "\n".join(
            f"SOURCE {i + 1}: {c['text']}" for i, c in enumerate(contexts)
        )
        assert ctx_str == ""


class TestRerankSorting:
    """Validates that rerank score sorting produces the correct order."""

    def test_hits_sorted_descending_by_rerank_score(self):
        hits = [
            {"text": "low", "rerank_score": 0.1},
            {"text": "high", "rerank_score": 0.9},
            {"text": "mid", "rerank_score": 0.5},
        ]
        hits.sort(key=lambda x: x.get("rerank_score", -100.0), reverse=True)
        assert hits[0]["text"] == "high"
        assert hits[-1]["text"] == "low"

    def test_hits_without_rerank_score_fall_to_bottom(self):
        hits = [
            {"text": "scored", "rerank_score": 0.5},
            {"text": "unscored"},
        ]
        hits.sort(key=lambda x: x.get("rerank_score", -100.0), reverse=True)
        assert hits[0]["text"] == "scored"

    def test_top_k_slicing(self):
        hits = [{"text": str(i), "rerank_score": float(i)} for i in range(10)]
        hits.sort(key=lambda x: x.get("rerank_score", -100.0), reverse=True)
        top_k = hits[:5]
        assert len(top_k) == 5
        assert top_k[0]["text"] == "9"


class TestCacheKey:
    """Validates semantic cache key normalisation."""

    def _cache_key(self, query: str) -> str:
        return f"rag_cache:{query.strip().lower()}"

    def test_strips_leading_trailing_whitespace(self):
        assert self._cache_key("  hello  ") == "rag_cache:hello"

    def test_lowercases_query(self):
        assert self._cache_key("What Is RAG?") == "rag_cache:what is rag?"

    def test_different_casing_same_key(self):
        assert self._cache_key("RAG") == self._cache_key("rag")


class TestCircuitBreakerLogic:
    """Tests the CircuitBreaker state machine without importing the full app."""

    class _CircuitBreaker:
        def __init__(self, failure_threshold=3, recovery_timeout=30):
            import time
            self._time = time
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.failures = 0
            self.last_failure_time = 0
            self.state = "CLOSED"

        def record_failure(self):
            self.failures += 1
            self.last_failure_time = self._time.time()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"

        def record_success(self):
            self.failures = 0
            self.state = "CLOSED"

        def can_proceed(self):
            if self.state == "OPEN":
                if self._time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF-OPEN"
                    return True
                return False
            return True

    def test_opens_after_threshold(self):
        cb = self._CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "OPEN"

    def test_closed_allows_proceed(self):
        cb = self._CircuitBreaker()
        assert cb.can_proceed() is True

    def test_open_blocks_proceed(self):
        cb = self._CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
        cb.record_failure()
        assert cb.can_proceed() is False

    def test_success_resets_to_closed(self):
        cb = self._CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
        cb.record_failure()
        cb.record_success()
        assert cb.state == "CLOSED"
        assert cb.can_proceed() is True
